import os
import json
from typing import Optional, Tuple

from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from dots_ocr.inference import inference_with_vllm
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import fitz_doc_to_image, load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md


class DotsOCRParser:
    """
    DotsOCR解析器：用于解析图像或PDF文件，提取布局和文本信息
    支持多种提示模式，可以处理不同类型的文档分析任务
    """

    def __init__(self,
                 ip='127.0.0.1',  # VLLM服务器IP地址
                 port=6006,  # VLLM服务器端口号
                 model_name='dots_ocr',  # 模型名称
                 temperature=0.1,  # 生成温度参数，控制随机性
                 top_p=1.0,  # 核采样参数
                 max_completion_tokens=16384,  # 最大完成标记数
                 num_thread=64,  # 处理PDF时的最大线程数
                 dpi=200,  # PDF处理时的DPI设置
                 output_dir="./output",  # 默认输出目录
                 min_pixels=None,  # 图像最小像素数限制
                 max_pixels=None,  # 图像最大像素数限制
                 use_hf=False,  # 是否使用HuggingFace模型而不是VLLM
                 model_path="./DotsOCR",  # HuggingFace模型路径（仅在use_hf=True时使用）
                 ):
        # 初始化图像处理参数
        self.dpi = dpi

        # VLLM服务器连接参数
        self.ip = ip
        self.port = port
        self.model_name = model_name

        # 模型推理参数
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread

        # 输出和图像处理参数
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # 模型类型选择
        self.use_hf = use_hf
        self.model_path = model_path
        if self.use_hf:
            # 使用HuggingFace模型时线程数设置为1
            print(f"使用HuggingFace模型，线程数将设置为1")
            self._load_hf_model()
        else:
            print(f"使用VLLM模型，线程数将设置为{self.num_thread}")
            self.model = None
            self.processor = None

        # 参数验证：确保像素数在合理范围内
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    def _load_hf_model(self):
        """
        加载 Huggingface Transformers 模型
        """
        import torch
        from transformers import AutoModelForCausalLM, Qwen2VLImageProcessor, AutoTokenizer
        from transformers.models.qwen2_vl import Qwen2VLProcessor
        
        print("=" * 60)
        print("正在加载 Huggingface Transformers 模型...")
        print(f"模型路径: {self.model_path}")
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("✓ 模型加载成功")
        
        # 使用 AutoProcessor 加载处理器（这样会自动加载正确的配置）
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        print("✓ Processor 加载成功")
        
        # 显示模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        device = next(self.model.parameters()).device
        print("=" * 60)
        print(f"模型信息:")
        print(f"  - 类型: {type(self.model).__name__}")
        print(f"  - 参数量: {total_params / 1e9:.2f}B")
        print(f"  - 设备: {device}")
        print(f"  - 精度: {next(self.model.parameters()).dtype}")
        print("=" * 60)

    def _inference_with_hf(self, image, prompt):
        """
        使用HuggingFace模型进行推理（按照官方示例）
        参数:
            image: 预处理后的图像 (PIL Image)
            prompt: 构造好的提示词
        返回:
            response: 模型响应结果
        """
        import torch
        from qwen_vl_utils import process_vision_info
        
        # 构建符合 DotsOCR 官方格式的消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image  # PIL Image 对象
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 按照官方示例处理输入
        # 1. 应用 chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 2. 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 3. 使用 processor 处理所有输入
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 将输入数据移动到模型所在设备
        inputs = inputs.to(self.model.device)

        # 模型推理生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_completion_tokens,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
            )

        # 后处理：修剪生成的标记（移除输入部分）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 解码生成结果
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response

    def _inference_with_vllm(self, image, prompt):
        """
        使用VLLM服务进行推理
        参数:
            image: 预处理后的图像
            prompt: 构造好的提示词
        返回:
            response: 模型响应结果
        """
        response = inference_with_vllm(
            image,
            prompt,
            model_name=self.model_name,
            ip=self.ip,
            port=self.port,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        """
        根据提示模式获取相应的提示词
        参数:
            prompt_mode: 提示模式名称
            bbox: 边界框坐标（用于目标检测模式）
            origin_image: 原始图像
            image: 预处理后的图像
            min_pixels: 最小像素数
            max_pixels: 最大像素数
        返回:
            prompt: 格式化后的提示词
        """
        # 从全局字典获取基础提示词
        prompt = dict_promptmode_to_prompt[prompt_mode]

        # 如果是目标检测模式，需要添加边界框信息
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None  # 必须提供边界框
            bboxes = [bbox]
            # 预处理边界框
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height,
                                      min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)  # 将边界框信息添加到提示词中
        return prompt

    def _parse_single_image(
            self,
            origin_image,
            prompt_mode,
            save_dir,
            save_name,
            source="image",
            page_idx=0,
            bbox=None,
            fitz_preprocess=False,
    ):
        """
        处理单张图像的核心方法
        参数:
            origin_image: 原始图像
            prompt_mode: 提示模式
            save_dir: 保存目录
            save_name: 保存文件名
            source: 来源类型（'image'或'pdf'）
            page_idx: 页面索引（用于PDF）
            bbox: 边界框坐标
            fitz_preprocess: 是否使用Fitz预处理
        返回:
            result: 包含解析结果的字典
        """
        # 设置像素限制
        min_pixels, max_pixels = self.min_pixels, self.max_pixels

        # 如果是目标检测模式，设置默认像素限制
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS
            max_pixels = max_pixels or MAX_PIXELS

        # 验证像素限制的合理性
        if min_pixels is not None:
            assert min_pixels >= MIN_PIXELS, f"最小像素数应 >= {MIN_PIXELS}"
        if max_pixels is not None:
            assert max_pixels <= MAX_PIXELS, f"最大像素数应 <= {MAX_PIXELS}"

        # 图像预处理
        if source == 'image' and fitz_preprocess:
            # 使用Fitz进行PDF图像预处理
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            # 普通图像预处理
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)

        # 智能调整图像尺寸
        input_height, input_width = smart_resize(image.height, image.width)

        # 获取提示词
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)

        # 根据模型类型选择推理方式
        if self.use_hf:
            response = self._inference_with_hf(image, prompt)
        else:
            response = self._inference_with_vllm(image, prompt)

        # 初始化结果字典
        result = {
            'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }

        # 如果是PDF页面，更新保存名称
        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"

        # 处理布局分析相关的提示模式
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            # 后处理模型输出
            cells, filtered = post_process_output(
                response,
                prompt_mode,
                origin_image,
                image,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

            # 如果输出被过滤（JSON解析失败），使用备用处理流程
            if filtered and prompt_mode != 'prompt_layout_only_en':
                # 保存原始响应JSON
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False)

                # 保存原始图像
                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)

                # 更新结果字典
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                # 保存Markdown内容
                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)

                result.update({
                    'md_content_path': md_file_path,
                    'filtered': True  # 标记为已过滤
                })
            else:
                # 正常处理流程：在图像上绘制布局
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"在图像上绘制布局时出错: {e}")
                    image_with_layout = origin_image

                # 保存布局信息JSON
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False)

                # 保存带布局的图像
                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)

                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                # 如果不是仅检测模式，生成Markdown内容
                if prompt_mode != "prompt_layout_only_en":
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True)

                    # 保存完整Markdown
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)

                    # 保存无页眉页脚的Markdown（用于评估）
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)

                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        else:
            # 处理其他提示模式（非布局分析）
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update({
                'layout_image_path': image_layout_path,
            })

            # 直接使用响应作为Markdown内容
            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        return result

    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        """
        解析单张图像文件
        参数:
            input_path: 输入文件路径
            filename: 文件名
            prompt_mode: 提示模式
            save_dir: 保存目录
            bbox: 边界框坐标
            fitz_preprocess: 是否使用Fitz预处理
        返回:
            包含解析结果的列表
        """
        # 获取原始图像
        origin_image = fetch_image(input_path)

        # 处理单张图像
        result = self._parse_single_image(
            origin_image,
            prompt_mode,
            save_dir,
            filename,
            source="image",
            bbox=bbox,
            fitz_preprocess=fitz_preprocess
        )

        # 添加文件路径信息
        result['file_path'] = input_path
        return [result]

    def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        """
        解析PDF文件
        参数:
            input_path: PDF文件路径
            filename: 文件名
            prompt_mode: 提示模式
            save_dir: 保存目录
        返回:
            包含所有页面解析结果的列表
        """
        print(f"正在加载PDF: {input_path}")

        # 从PDF加载所有图像
        images_origin = load_images_from_pdf(input_path, dpi=self.dpi)
        total_pages = len(images_origin)

        # 创建处理任务列表
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source": "pdf",
                "page_idx": i,
            } for i, image in enumerate(images_origin)
        ]

        # 定义任务执行函数
        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        # 根据模型类型设置线程数
        if self.use_hf:
            num_thread = 1
        else:
            num_thread = min(total_pages, self.num_thread)

        print(f"使用{num_thread}个线程处理{total_pages}页PDF...")

        # 使用线程池并行处理页面
        results = []
        with ThreadPool(num_thread) as pool:
            with tqdm(total=total_pages, desc="处理PDF页面") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    results.append(result)
                    pbar.update(1)

        # 按页面序号排序结果
        results.sort(key=lambda x: x["page_no"])

        # 为每个结果添加文件路径信息
        for i in range(len(results)):
            results[i]['file_path'] = input_path

        return results

    def parse_file(self,
                   input_path,
                   output_dir="",
                   prompt_mode="prompt_layout_all_en",
                   bbox=None,
                   fitz_preprocess=False
                   ):
        """
        解析文件（自动识别PDF或图像）
        参数:
            input_path: 输入文件路径
            output_dir: 输出目录
            prompt_mode: 提示模式
            bbox: 边界框坐标
            fitz_preprocess: 是否使用Fitz预处理
        返回:
            解析结果列表
        """
        # 设置输出目录
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)

        # 创建保存目录
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        # 根据文件扩展选择处理方式
        if file_ext == '.pdf':
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir)
        elif file_ext in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox,
                                       fitz_preprocess=fitz_preprocess)
        else:
            raise ValueError(f"不支持的文件扩展名 {file_ext}，支持的扩展名包括 {image_extensions} 和 pdf")

        # 保存所有结果到JSONL文件
        print(f"解析完成，结果保存到 {save_dir}")
        with open(os.path.join(output_dir, os.path.basename(filename) + '.jsonl'), 'w', encoding="utf-8") as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

        return results


def do_parse(
        input_path: str,
        output: str = "/home/lf/mount/LLM/project/graph_rag_deom/dots_ocr/output",
        prompt: str = "prompt_layout_all_en",
        bbox: Optional[Tuple[int, int, int, int]] = None,
        ip: str = "localhost",
        port: int = 6006,
        model_name: str = "dots_ocr",
        temperature: float = 0.1,
        top_p: float = 1.0,
        dpi: int = 200,
        max_completion_tokens: int = 16384,
        num_thread: int = 16,
        no_fitz_preprocess: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        use_hf: bool = False,
        model_path: str = "/home/lf/mount/LLM/project/graph_rag_deom/DotsOCR"
):
    """
    dots.ocr 多语言文档布局解析器

    参数:
        input_path (str): 输入PDF/图像文件路径
        output (str): 输出目录 (默认: ./output)
        prompt (str): 用于查询模型的提示词，不同任务使用不同的提示词
        bbox (Optional[Tuple[int, int, int, int]]): 边界框坐标 (x1, y1, x2, y2)
        ip (str): 服务器IP地址 (默认: localhost)
        port (int): 服务器端口 (默认: 8000)
        model_name (str): 模型名称 (默认: model)
        temperature (float): 温度参数 (默认: 0.1)
        top_p (float): 核采样参数 (默认: 1.0)
        dpi (int): DPI设置 (默认: 200)
        max_completion_tokens (int): 最大完成标记数 (默认: 16384)
        num_thread (int): 线程数 (默认: 16)
        no_fitz_preprocess (bool): 是否禁用Fitz预处理 (默认: False)指的是选择是否使用PyMuPDF（fitz）库对图像输入进行特定的预处理操作
        min_pixels (Optional[int]): 最小像素数
        max_pixels (Optional[int]): 最大像素数
        use_hf (bool): 是否使用HuggingFace (默认: False)
        model_path (str): Huggingface模型路径 (仅在use_hf=True时使用，默认: /home/lf/mount/LLM/project/graph_rag_deom/DotsOCR)
    """
    # 获取所有可用的提示模式
    prompts = list(dict_promptmode_to_prompt.keys())

    # 验证prompt参数是否有效
    if prompt not in prompts:
        raise ValueError(f"无效的prompt参数: {prompt}。可选值: {prompts}")

    # 创建DotsOCR解析器实例
    dots_ocr_parser = DotsOCRParser(
        ip=ip,
        port=port,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        num_thread=num_thread,
        dpi=dpi,
        output_dir=output,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_hf=use_hf,
        model_path=model_path,
    )

    # 设置Fitz预处理标志
    fitz_preprocess = not no_fitz_preprocess
    if fitz_preprocess:
        print(f"对图像输入使用Fitz预处理，请检查图像像素的变化")

    # 解析文件
    result = dots_ocr_parser.parse_file(
        input_path,
        prompt_mode=prompt,
        bbox=bbox,
        fitz_preprocess=fitz_preprocess,
    )

    return result


if __name__ == "__main__":
    # main()
    # do_parse(input_path='../demo_image1.jpg')
    # do_parse(input_path='../demo_pdf1.pdf', num_thread=32)
    
    # 使用 HuggingFace Transformers 模型
    do_parse(
        input_path='/home/lf/mount/LLM/project/graph_rag_deom/第一章 Apache Flink 概述.pdf', 
        use_hf=True,
        model_path='/home/lf/mount/LLM/project/MutilModel_RAG/DotsOCR'
    )
