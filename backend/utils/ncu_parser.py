#!/usr/bin/env python3
"""
NVIDIA Nsight Compute (ncu) 输出文件自动化解析工具

支持解析多种 ncu 输出格式：
- NCU Report 文件 (.ncu-rep)
- CSV 导出文件
- JSON 导出文件
- 自动调用 ncu 导出工具

专注于CUDA kernel级别的详细性能分析，包括：
- GPU 利用率指标
- 内存带宽分析  
- Warp 执行效率
- 指令吞吐量
- 占用率分析

作者: AI助手
版本: 1.0
"""

import os
import sys
import json
import csv
from io import StringIO
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

INTEGRATED_MD = Path("/workspace/Agent/AI_Agent_Complete/sglang_analysis_b8_i512_o64/integrated_performance_report.md")

@dataclass
class KernelMetrics:
    """CUDA Kernel 性能指标"""
    name: str
    grid_size: Optional[Tuple[int, int, int]] = None
    block_size: Optional[Tuple[int, int, int]] = None
    
    # GPU 利用率指标
    sm_efficiency: Optional[float] = None  # SM效率 (%)
    achieved_occupancy: Optional[float] = None  # 实现占用率 (%)
    theoretical_occupancy: Optional[float] = None  # 理论占用率 (%)
    
    # 内存性能指标
    dram_bandwidth: Optional[float] = None  # DRAM带宽 (GB/s)
    l2_hit_rate: Optional[float] = None  # L2缓存命中率 (%)
    l1_hit_rate: Optional[float] = None  # L1缓存命中率 (%)
    
    # 计算性能指标
    tensor_active: Optional[float] = None  # Tensor Core活跃度 (%)
    fp32_pipe_utilization: Optional[float] = None  # FP32管道利用率 (%)
    fp16_pipe_utilization: Optional[float] = None  # FP16管道利用率 (%)
    int_pipe_utilization: Optional[float] = None  # INT管道利用率 (%)
    
    # Warp 执行指标
    warp_execution_efficiency: Optional[float] = None  # Warp执行效率 (%)
    warp_stall_long_scoreboard: Optional[float] = None  # 长记分板停顿 (%)
    warp_stall_memory_throttle: Optional[float] = None  # 内存限流停顿 (%)
    warp_stall_memory_dependency: Optional[float] = None  # 内存依赖停顿 (%)
    
    # 其他指标
    duration: Optional[float] = None  # 执行时间 (ms)
    registers_per_thread: Optional[int] = None
    shared_memory_per_block: Optional[int] = None
    
@dataclass 
class BottleneckInfo:
    """性能瓶颈信息"""
    type: str  # 瓶颈类型: memory, compute, latency
    severity: str  # 严重程度: low, medium, high, critical
    description: str
    metrics: Dict[str, float]
    recommendations: List[str]

class NCUParser:
    """NCU 输出文件解析器

    统一输出路径策略:
        所有导出的 CSV / JSON 均写到 /workspace/Agent/AI_Agent_Complete 下，
        便于 Agent 聚合读取。
    """
    DEFAULT_BASE_DIR = Path("/workspace/Agent/AI_Agent_Complete")

    def __init__(self, input_file: str):
        self.input_file = Path(input_file)
        # 标准化为绝对路径下的文件（如果给的是相对路径）
        if not self.input_file.is_absolute():
            self.input_file = Path.cwd() / self.input_file
        self.kernels: List[KernelMetrics] = []
        self.raw_data: Dict = {}
        self.metadata: Dict = {}
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    def parse(self) -> None:
        """解析输入文件"""
        suffix = self.input_file.suffix.lower()
        
        if suffix == '.ncu-rep':
            self._parse_ncu_rep()
        elif suffix == '.csv':
            self._parse_csv()
        elif suffix == '.json':
            self._parse_json()
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def _parse_ncu_rep(self) -> None:
        """解析 .ncu-rep 文件（需要先导出为CSV）"""
        print("📋 检测到 .ncu-rep 文件，正在导出为CSV格式...")
        
        # 生成输出文件名
        # 强制输出到统一目录
        csv_file = self.DEFAULT_BASE_DIR / (self.input_file.stem + '.csv')
        
        # 调用 ncu 导出命令
        cmd = [
            'ncu', '--csv',
            '--log-file', str(csv_file),
            '--import', str(self.input_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ 导出成功: {csv_file}")
            
            # 解析导出的CSV文件
            self._parse_csv(csv_file)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ncu导出失败: {e.stderr}")
            print("尝试使用替代导出方式...")
            # 尝试JSON格式导出
            self._parse_ncu_rep_json()
            
        except FileNotFoundError:
            print("❌ 未找到 ncu 命令")
            print("请安装 NVIDIA Nsight Compute 并确保 ncu 在PATH中")
            raise
    
    def _parse_ncu_rep_json(self) -> None:
        """解析 .ncu-rep 文件导出为JSON"""
        json_file = self.DEFAULT_BASE_DIR / (self.input_file.stem + '.json')
        
        cmd = [
            'ncu', '--json',
            '--log-file', str(json_file), 
            '--import', str(self.input_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ JSON导出成功: {json_file}")
            self._parse_json(json_file)
        except subprocess.CalledProcessError as e:
            print(f"❌ JSON导出也失败: {e.stderr}")
            raise
    
    def _parse_csv(self, csv_file: Optional[Path] = None) -> None:
        """解析 CSV 文件"""
        target_file = csv_file or self.input_file
        print(f"📊 正在解析CSV文件: {target_file}")
        path_obj = Path(target_file)
        if not path_obj.exists():
            print("⚠️ CSV文件不存在，跳过")
            return
        if path_obj.stat().st_size == 0:
            print("⚠️ CSV为空，尝试 JSON 回退...")
            self._fallback_to_json()
            return
        try:
            # 预清洗：过滤注释和空行
            with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [l for l in f.readlines() if l.strip() and not l.startswith('#')]
            if not lines:
                print("⚠️ 有效内容为空，尝试 JSON 回退...")
                self._fallback_to_json()
                return
            temp_clean = path_obj.parent / (path_obj.stem + '.csv')
            temp_clean.write_text('\n'.join(lines), encoding='utf-8')
            df = pd.read_csv(temp_clean)
            # cleaned_csv_text = '\n'.join(lines)
            # df = pd.read_csv(StringIO(cleaned_csv_text))
            print(f"🔍 发现 {len(df)} 行数据，列: {list(df.columns)}")
            if df.empty:
                print("⚠️ DataFrame为空，尝试 JSON 回退...")
                self._fallback_to_json()
                return
            if 'Kernel Name' in df.columns or 'KernelName' in df.columns:
                self._parse_csv_kernels(df)
            else:
                self._parse_csv_generic(df)
        except Exception as e:
            print(f"❌ CSV解析失败: {e} -> 尝试 JSON 回退")
            self._fallback_to_json()

    def _fallback_to_json(self) -> None:
        """当 CSV 解析失败或为空时回退至 JSON 导出解析"""
        try:
            self._parse_ncu_rep_json()
        except Exception as e:
            print(f"⚠️ JSON 回退也失败: {e}")
    
    def _parse_csv_kernels(self, df: pd.DataFrame) -> None:
        """解析包含kernel信息的CSV"""
        kernel_name_col = 'Kernel Name' if 'Kernel Name' in df.columns else 'KernelName'

        # 新增：支持“长表”结构（每行=一个指标）
        if {'Section Name', 'Metric Name', 'Metric Value'}.issubset(df.columns):
            # 规范化 Metric Value 为数值
            df['Metric Value'] = df['Metric Value'].astype(str).str.replace(',', '', regex=False)
            df['Metric Value'] = pd.to_numeric(df['Metric Value'], errors='coerce')

            for kname, kdf in df.groupby(kernel_name_col):
                metrics = KernelMetrics(name=str(kname))

                def get_metric(section: str, name: str):
                    sel = kdf[(kdf['Section Name'] == section) & (kdf['Metric Name'] == name)]['Metric Value']
                    return None if sel.empty else float(sel.mean())

                # 网格/块
                try:
                    bsz = str(kdf['Block Size'].dropna().iloc[0])
                    gsz = str(kdf['Grid Size'].dropna().iloc[0])
                    def parse_xyz(s):
                        s = s.strip().strip('()')
                        x, y, z = [int(float(v.strip())) for v in s.split(',')]
                        return (x, y, z)
                    metrics.block_size = parse_xyz(bsz)
                    metrics.grid_size  = parse_xyz(gsz)
                except Exception:
                    pass

                # 常见映射
                # SM Busy（或用 GPU SOL Throughput 的 Compute(SM) 指标作为近似）
                sm_busy = get_metric('Compute Workload Analysis', 'SM Busy')
                if sm_busy is None:
                    sm_busy = get_metric('GPU Speed Of Light Throughput', 'Compute (SM) Throughput')
                metrics.sm_efficiency = sm_busy

                # Occupancy
                metrics.achieved_occupancy    = get_metric('Occupancy', 'Achieved Occupancy')
                metrics.theoretical_occupancy = get_metric('Occupancy', 'Theoretical Occupancy')

                # Memory
                # 不同版本列名可能为 "Memory Throughput" 或 "DRAM Throughput"
                m_bw = get_metric('Memory Workload Analysis', 'Memory Throughput')
                if m_bw is None:
                    m_bw = get_metric('Memory Workload Analysis', 'DRAM Throughput')
                metrics.dram_bandwidth = m_bw
                metrics.l2_hit_rate    = get_metric('Memory Workload Analysis', 'L2 Hit Rate')
                # L1/TEX 命中率（有的版本叫 L1/TEX Hit Rate）
                l1_rate = get_metric('Memory Workload Analysis', 'L1/TEX Hit Rate')
                if l1_rate is None:
                    l1_rate = get_metric('Memory Workload Analysis', 'L1 Hit Rate')
                metrics.l1_hit_rate = l1_rate

                # Duration（单位可能是 us，保守不换算；如需 ms 可除以 1000）
                dur = get_metric('GPU Speed Of Light Throughput', 'Duration')
                if dur is not None:
                    metrics.duration = dur  # 如需 ms: dur/1000.0

                # Launch Statistics
                regs = get_metric('Launch Statistics', 'Registers Per Thread')
                if regs is not None:
                    metrics.registers_per_thread = int(regs)

                shm_dyn_kb = get_metric('Launch Statistics', 'Dynamic Shared Memory Per Block')
                shm_sta_b  = get_metric('Launch Statistics', 'Static Shared Memory Per Block')
                if shm_dyn_kb is not None or shm_sta_b is not None:
                    dyn_b = (shm_dyn_kb or 0) * 1024.0
                    sta_b = (shm_sta_b or 0)
                    metrics.shared_memory_per_block = int(dyn_b + sta_b)

                self.kernels.append(metrics)

            print(f"🔥 解析到 {len(self.kernels)} 个kernel（长表）")
            return

        # 原有“宽表”解析（保留）
        for kernel_name in df[kernel_name_col].unique():
            kernel_data = df[df[kernel_name_col] == kernel_name].iloc[0]
            metrics = KernelMetrics(name=kernel_name)
            column_mapping = {
                'SM Efficiency': 'sm_efficiency',
                'Achieved Occupancy': 'achieved_occupancy',
                'Theoretical Occupancy': 'theoretical_occupancy',
                'DRAM Bandwidth': 'dram_bandwidth',
                'L2 Hit Rate': 'l2_hit_rate',
                'L1 Hit Rate': 'l1_hit_rate',
                'Tensor Active': 'tensor_active',
                'FP32 Pipeline Utilization': 'fp32_pipe_utilization',
                'Warp Execution Efficiency': 'warp_execution_efficiency',
                'Duration': 'duration',
                'Registers Per Thread': 'registers_per_thread',
                'Grid Size': 'grid_size',
                'Block Size': 'block_size'
            }
            for col_name, attr_name in column_mapping.items():
                if col_name in df.columns:
                    value = kernel_data[col_name]
                    if pd.notna(value):
                        setattr(metrics, attr_name, value)
            self.kernels.append(metrics)
        print(f"🔥 解析到 {len(self.kernels)} 个kernels")
    
    def _parse_csv_generic(self, df: pd.DataFrame) -> None:
        """解析通用CSV格式"""
        # 假设每行是一个kernel的数据
        for _, row in df.iterrows():
            metrics = KernelMetrics(name=f"Kernel_{len(self.kernels)}")
            
            # 尝试从列名推断指标
            for col_name, value in row.items():
                if pd.isna(value):
                    continue
                    
                col_lower = col_name.lower()
                if 'sm' in col_lower and 'efficiency' in col_lower:
                    metrics.sm_efficiency = float(value)
                elif 'occupancy' in col_lower:
                    if 'achieved' in col_lower:
                        metrics.achieved_occupancy = float(value)
                    elif 'theoretical' in col_lower:
                        metrics.theoretical_occupancy = float(value)
                elif 'bandwidth' in col_lower:
                    metrics.dram_bandwidth = float(value)
                elif 'duration' in col_lower:
                    metrics.duration = float(value)
            
            self.kernels.append(metrics)
        
        print(f"🔥 解析到 {len(self.kernels)} 个kernels")
    
    def _parse_json(self, json_file: Optional[Path] = None) -> None:
        """解析 JSON 文件"""
        target_file = json_file or self.input_file
        print(f"📋 正在解析JSON文件: {target_file}")
        
        with open(target_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.raw_data = data
        
        # 根据JSON结构解析
        if isinstance(data, list):
            self._parse_json_list(data)
        elif isinstance(data, dict):
            self._parse_json_dict(data)
    
    def _parse_json_list(self, data: List) -> None:
        """解析JSON列表格式"""
        for item in data:
            if isinstance(item, dict):
                metrics = self._extract_metrics_from_dict(item)
                if metrics:
                    self.kernels.append(metrics)
    
    def _parse_json_dict(self, data: Dict) -> None:
        """解析JSON字典格式"""
        if 'kernels' in data:
            self._parse_json_list(data['kernels'])
        elif 'reports' in data:
            for report in data['reports']:
                if 'kernels' in report:
                    self._parse_json_list(report['kernels'])
    
    def _extract_metrics_from_dict(self, data: Dict) -> Optional[KernelMetrics]:
        """从字典中提取性能指标"""
        if 'name' not in data and 'kernel' not in data:
            return None
        
        name = data.get('name', data.get('kernel', f"Kernel_{len(self.kernels)}"))
        metrics = KernelMetrics(name=name)
        
        # 映射JSON字段到指标
        field_mapping = {
            'smEfficiency': 'sm_efficiency',
            'achievedOccupancy': 'achieved_occupancy',
            'theoreticalOccupancy': 'theoretical_occupancy', 
            'dramBandwidth': 'dram_bandwidth',
            'l2HitRate': 'l2_hit_rate',
            'l1HitRate': 'l1_hit_rate',
            'tensorActive': 'tensor_active',
            'warpExecutionEfficiency': 'warp_execution_efficiency',
            'duration': 'duration',
            'registersPerThread': 'registers_per_thread'
        }
        
        for json_field, attr_name in field_mapping.items():
            if json_field in data:
                setattr(metrics, attr_name, data[json_field])
        
        return metrics

class NCUAnalyzer:
    """NCU 数据分析器"""
    
    def __init__(self, parser: NCUParser):
        self.parser = parser
        self.stats = {}
        self.bottlenecks: List[BottleneckInfo] = []
    
    def analyze(self) -> Dict:
        """执行完整分析"""
        print("🔍 开始NCU性能分析...")
        
        self.stats = {
            'gpu_utilization': self._analyze_gpu_utilization(),
            'memory_analysis': self._analyze_memory_performance(), 
            'compute_analysis': self._analyze_compute_performance(),
            'warp_analysis': self._analyze_warp_efficiency(),
            'occupancy_analysis': self._analyze_occupancy(),
            'bottleneck_analysis': self._identify_bottlenecks()
        }
        
        return self.stats
    
    def _analyze_gpu_utilization(self) -> Dict:
        """分析GPU利用率"""
        if not self.parser.kernels:
            return {}
        
        sm_efficiencies = [k.sm_efficiency for k in self.parser.kernels if k.sm_efficiency is not None]
        
        if not sm_efficiencies:
            return {'message': '无SM效率数据'}
        
        return {
            'average_sm_efficiency': sum(sm_efficiencies) / len(sm_efficiencies),
            'max_sm_efficiency': max(sm_efficiencies),
            'min_sm_efficiency': min(sm_efficiencies),
            'kernels_below_50_percent': len([x for x in sm_efficiencies if x < 50]),
            'total_kernels': len(sm_efficiencies)
        }
    
    def _analyze_memory_performance(self) -> Dict:
        """分析内存性能"""
        bandwidth_data = [k.dram_bandwidth for k in self.parser.kernels if k.dram_bandwidth is not None]
        l2_hit_rates = [k.l2_hit_rate for k in self.parser.kernels if k.l2_hit_rate is not None]
        l1_hit_rates = [k.l1_hit_rate for k in self.parser.kernels if k.l1_hit_rate is not None]
        
        analysis = {}
        
        if bandwidth_data:
            analysis['bandwidth_stats'] = {
                'average_bandwidth': sum(bandwidth_data) / len(bandwidth_data),
                'max_bandwidth': max(bandwidth_data),
                'min_bandwidth': min(bandwidth_data)
            }
        
        if l2_hit_rates:
            analysis['l2_cache_stats'] = {
                'average_l2_hit_rate': sum(l2_hit_rates) / len(l2_hit_rates),
                'kernels_low_l2_hit_rate': len([x for x in l2_hit_rates if x < 50])
            }
        
        if l1_hit_rates:
            analysis['l1_cache_stats'] = {
                'average_l1_hit_rate': sum(l1_hit_rates) / len(l1_hit_rates),
                'kernels_low_l1_hit_rate': len([x for x in l1_hit_rates if x < 50])
            }
        
        return analysis
    
    def _analyze_compute_performance(self) -> Dict:
        """分析计算性能"""
        tensor_active = [k.tensor_active for k in self.parser.kernels if k.tensor_active is not None]
        fp32_util = [k.fp32_pipe_utilization for k in self.parser.kernels if k.fp32_pipe_utilization is not None]
        
        analysis = {}
        
        if tensor_active:
            analysis['tensor_core_usage'] = {
                'average_tensor_active': sum(tensor_active) / len(tensor_active),
                'kernels_using_tensor': len([x for x in tensor_active if x > 0])
            }
        
        if fp32_util:
            analysis['fp32_pipeline'] = {
                'average_fp32_utilization': sum(fp32_util) / len(fp32_util),
                'max_fp32_utilization': max(fp32_util)
            }
        
        return analysis
    
    def _analyze_warp_efficiency(self) -> Dict:
        """分析Warp执行效率"""
        warp_eff = [k.warp_execution_efficiency for k in self.parser.kernels if k.warp_execution_efficiency is not None]
        
        if not warp_eff:
            return {'message': '无Warp效率数据'}
        
        return {
            'average_warp_efficiency': sum(warp_eff) / len(warp_eff),
            'min_warp_efficiency': min(warp_eff),
            'kernels_low_warp_efficiency': len([x for x in warp_eff if x < 70])
        }
    
    def _analyze_occupancy(self) -> Dict:
        """分析占用率"""
        achieved_occ = [k.achieved_occupancy for k in self.parser.kernels if k.achieved_occupancy is not None]
        theoretical_occ = [k.theoretical_occupancy for k in self.parser.kernels if k.theoretical_occupancy is not None]
        
        analysis = {}
        
        if achieved_occ:
            analysis['achieved_occupancy'] = {
                'average': sum(achieved_occ) / len(achieved_occ),
                'min': min(achieved_occ),
                'max': max(achieved_occ)
            }
        
        if theoretical_occ:
            analysis['theoretical_occupancy'] = {
                'average': sum(theoretical_occ) / len(theoretical_occ),
                'min': min(theoretical_occ),
                'max': max(theoretical_occ)
            }
        
        if achieved_occ and theoretical_occ and len(achieved_occ) == len(theoretical_occ):
            occupancy_ratios = [a/t if t > 0 else 0 for a, t in zip(achieved_occ, theoretical_occ)]
            analysis['occupancy_efficiency'] = {
                'average_ratio': sum(occupancy_ratios) / len(occupancy_ratios),
                'kernels_low_efficiency': len([x for x in occupancy_ratios if x < 0.8])
            }
        
        return analysis
    
    def _identify_bottlenecks(self) -> Dict:
        """识别性能瓶颈"""
        self.bottlenecks.clear()
        
        for kernel in self.parser.kernels:
            kernel_bottlenecks = []
            
            # 检查SM效率
            if kernel.sm_efficiency is not None and kernel.sm_efficiency < 30:
                kernel_bottlenecks.append(BottleneckInfo(
                    type="compute",
                    severity="high" if kernel.sm_efficiency < 15 else "medium",
                    description=f"SM效率过低 ({kernel.sm_efficiency:.1f}%)",
                    metrics={"sm_efficiency": kernel.sm_efficiency},
                    recommendations=["检查kernel算法复杂度", "考虑增加工作负载"]
                ))
            
            # 检查内存带宽
            if kernel.dram_bandwidth is not None and kernel.dram_bandwidth < 100:
                kernel_bottlenecks.append(BottleneckInfo(
                    type="memory",
                    severity="medium",
                    description=f"内存带宽利用率低 ({kernel.dram_bandwidth:.1f} GB/s)",
                    metrics={"dram_bandwidth": kernel.dram_bandwidth},
                    recommendations=["优化内存访问模式", "考虑合并访问"]
                ))
            
            # 检查缓存命中率
            if kernel.l2_hit_rate is not None and kernel.l2_hit_rate < 70:
                kernel_bottlenecks.append(BottleneckInfo(
                    type="memory", 
                    severity="medium",
                    description=f"L2缓存命中率低 ({kernel.l2_hit_rate:.1f}%)",
                    metrics={"l2_hit_rate": kernel.l2_hit_rate},
                    recommendations=["改善数据局部性", "减少不规则内存访问"]
                ))
            
            # 检查占用率
            if (kernel.achieved_occupancy is not None and 
                kernel.theoretical_occupancy is not None and
                kernel.theoretical_occupancy > 0):
                
                occupancy_ratio = kernel.achieved_occupancy / kernel.theoretical_occupancy
                if occupancy_ratio < 0.7:
                    kernel_bottlenecks.append(BottleneckInfo(
                        type="latency",
                        severity="medium",
                        description=f"占用率效率低 ({occupancy_ratio*100:.1f}%)",
                        metrics={
                            "achieved_occupancy": kernel.achieved_occupancy,
                            "theoretical_occupancy": kernel.theoretical_occupancy
                        },
                        recommendations=["检查资源限制", "优化寄存器使用", "优化共享内存使用"]
                    ))
            
            self.bottlenecks.extend(kernel_bottlenecks)
        
        # 分析瓶颈统计
        bottleneck_stats = {
            'total_bottlenecks': len(self.bottlenecks),
            'bottleneck_types': {},
            'severity_distribution': {},
            'top_issues': []
        }
        
        for bottleneck in self.bottlenecks:
            # 统计类型
            if bottleneck.type not in bottleneck_stats['bottleneck_types']:
                bottleneck_stats['bottleneck_types'][bottleneck.type] = 0
            bottleneck_stats['bottleneck_types'][bottleneck.type] += 1
            
            # 统计严重程度
            if bottleneck.severity not in bottleneck_stats['severity_distribution']:
                bottleneck_stats['severity_distribution'][bottleneck.severity] = 0
            bottleneck_stats['severity_distribution'][bottleneck.severity] += 1
        
        # 获取主要问题
        bottleneck_stats['top_issues'] = [
            {
                'description': b.description,
                'type': b.type,
                'severity': b.severity,
                'recommendations': b.recommendations[:2]  # 只显示前2个建议
            }
            for b in sorted(self.bottlenecks, 
                          key=lambda x: {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(x.severity, 0),
                          reverse=True)[:5]
        ]
        
        return bottleneck_stats

class NCUVisualizer:
    """NCU 数据可视化"""
    
    def __init__(self, parser: NCUParser, analyzer: NCUAnalyzer):
        self.parser = parser
        self.analyzer = analyzer
        self.output_dir = Path("ncu_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_visualizations(self) -> None:
        """创建所有可视化图表"""
        print("📊 生成NCU可视化图表...")
        
        if self.parser.kernels:
            self._plot_gpu_utilization()
            self._plot_memory_performance()
            self._plot_occupancy_analysis()
            self._plot_bottleneck_analysis()
            self._plot_kernel_comparison()
        
        print(f"📁 图表已保存到: {self.output_dir}")
    
    def _plot_gpu_utilization(self) -> None:
        """绘制GPU利用率分析"""
        sm_efficiencies = [(k.name, k.sm_efficiency) for k in self.parser.kernels 
                          if k.sm_efficiency is not None]
        
        if not sm_efficiencies:
            return
        
        names, efficiencies = zip(*sm_efficiencies)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # SM效率柱状图
        colors = ['red' if eff < 30 else 'orange' if eff < 60 else 'green' 
                 for eff in efficiencies]
        
        ax1.bar(range(len(names)), efficiencies, color=colors)
        ax1.set_xlabel('Kernel 索引')
        ax1.set_ylabel('SM 效率 (%)')
        ax1.set_title('各Kernel SM效率')
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50%基线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 效率分布直方图
        ax2.hist(efficiencies, bins=20, alpha=0.7, color='skyblue')
        ax2.set_xlabel('SM 效率 (%)')
        ax2.set_ylabel('Kernel 数量')
        ax2.set_title('SM效率分布')
        ax2.axvline(x=sum(efficiencies)/len(efficiencies), color='red', 
                   linestyle='--', label=f'平均值: {sum(efficiencies)/len(efficiencies):.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gpu_utilization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_performance(self) -> None:
        """绘制内存性能分析"""
        # 收集内存相关数据
        bandwidth_data = [(k.name, k.dram_bandwidth) for k in self.parser.kernels 
                         if k.dram_bandwidth is not None]
        l2_hit_rates = [(k.name, k.l2_hit_rate) for k in self.parser.kernels 
                       if k.l2_hit_rate is not None]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # DRAM带宽
        if bandwidth_data:
            names, bandwidths = zip(*bandwidth_data)
            axes[0, 0].bar(range(len(names)), bandwidths, color='lightcoral')
            axes[0, 0].set_title('DRAM 带宽利用率')
            axes[0, 0].set_ylabel('带宽 (GB/s)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # L2命中率
        if l2_hit_rates:
            names, rates = zip(*l2_hit_rates)
            colors = ['red' if rate < 50 else 'orange' if rate < 80 else 'green' 
                     for rate in rates]
            axes[0, 1].bar(range(len(names)), rates, color=colors)
            axes[0, 1].set_title('L2 缓存命中率')
            axes[0, 1].set_ylabel('命中率 (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 带宽分布
        if bandwidth_data:
            _, bandwidths = zip(*bandwidth_data)
            axes[1, 0].hist(bandwidths, bins=15, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('带宽分布')
            axes[1, 0].set_xlabel('带宽 (GB/s)')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 缓存命中率分布
        if l2_hit_rates:
            _, rates = zip(*l2_hit_rates)
            axes[1, 1].hist(rates, bins=15, alpha=0.7, color='lightyellow')
            axes[1, 1].set_title('L2命中率分布')
            axes[1, 1].set_xlabel('命中率 (%)')
            axes[1, 1].set_ylabel('频次')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_occupancy_analysis(self) -> None:
        """绘制占用率分析"""
        occupancy_data = []
        for k in self.parser.kernels:
            if k.achieved_occupancy is not None and k.theoretical_occupancy is not None:
                occupancy_data.append({
                    'name': k.name,
                    'achieved': k.achieved_occupancy,
                    'theoretical': k.theoretical_occupancy,
                    'ratio': k.achieved_occupancy / k.theoretical_occupancy if k.theoretical_occupancy > 0 else 0
                })
        
        if not occupancy_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 占用率对比
        names = [d['name'][:20] + '...' if len(d['name']) > 20 else d['name'] for d in occupancy_data]
        achieved = [d['achieved'] for d in occupancy_data]
        theoretical = [d['theoretical'] for d in occupancy_data]
        
        x = range(len(names))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], achieved, width, label='实际占用率', color='lightblue')
        ax1.bar([i + width/2 for i in x], theoretical, width, label='理论占用率', color='lightcoral')
        
        ax1.set_xlabel('Kernel')
        ax1.set_ylabel('占用率 (%)')
        ax1.set_title('占用率对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 占用率效率分布
        ratios = [d['ratio'] * 100 for d in occupancy_data]
        ax2.hist(ratios, bins=15, alpha=0.7, color='lightgreen')
        ax2.set_xlabel('占用率效率 (%)')
        ax2.set_ylabel('Kernel 数量')
        ax2.set_title('占用率效率分布')
        ax2.axvline(x=80, color='red', linestyle='--', alpha=0.7, label='80%基线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'occupancy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bottleneck_analysis(self) -> None:
        """绘制瓶颈分析"""
        if not self.analyzer.bottlenecks:
            return
        
        # 统计瓶颈类型
        bottleneck_types = {}
        severity_counts = {}
        
        for b in self.analyzer.bottlenecks:
            bottleneck_types[b.type] = bottleneck_types.get(b.type, 0) + 1
            severity_counts[b.severity] = severity_counts.get(b.severity, 0) + 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 瓶颈类型分布
        types = list(bottleneck_types.keys())
        counts = list(bottleneck_types.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        ax1.pie(counts, labels=types, autopct='%1.1f%%', colors=colors[:len(types)])
        ax1.set_title('性能瓶颈类型分布')
        
        # 严重程度分布
        severities = list(severity_counts.keys())
        severity_colors = {
            'critical': '#ff4444',
            'high': '#ff8844',
            'medium': '#ffcc44', 
            'low': '#88cc88'
        }
        bar_colors = [severity_colors.get(s, '#cccccc') for s in severities]
        
        ax2.bar(severities, [severity_counts[s] for s in severities], color=bar_colors)
        ax2.set_title('瓶颈严重程度分布')
        ax2.set_ylabel('数量')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bottleneck_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_kernel_comparison(self) -> None:
        """绘制kernel性能对比雷达图"""
        if len(self.parser.kernels) < 2:
            return
        
        # 选择前几个有完整数据的kernel
        complete_kernels = []
        for k in self.parser.kernels:
            if (k.sm_efficiency is not None and 
                k.achieved_occupancy is not None and
                k.dram_bandwidth is not None):
                complete_kernels.append(k)
                if len(complete_kernels) >= 5:  # 最多显示5个
                    break
        
        if len(complete_kernels) < 2:
            return
        
        # 准备雷达图数据
        metrics = ['SM效率', '占用率', 'DRAM带宽', 'L2命中率', 'Warp效率']
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = [n / float(len(metrics)) * 2 * 3.14159 for n in range(len(metrics))]
        angles += angles[:1]
        
        colors = plt.cm.Set3(range(len(complete_kernels)))
        
        for i, kernel in enumerate(complete_kernels):
            values = [
                kernel.sm_efficiency or 0,
                kernel.achieved_occupancy or 0,
                min(kernel.dram_bandwidth or 0, 100) if kernel.dram_bandwidth else 0,  # 归一化到100
                kernel.l2_hit_rate or 0,
                kernel.warp_execution_efficiency or 0
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=kernel.name[:20], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 100)
        ax.set_title('Kernel 性能对比雷达图')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kernel_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

class NCUReporter:
    """NCU 分析报告生成器"""
    
    def __init__(self, parser: NCUParser, analyzer: NCUAnalyzer):
        self.parser = parser
        self.analyzer = analyzer
        self.output_dir = Path("ncu_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(self) -> None:
        """生成分析报告"""
        print("📄 生成NCU分析报告...")
        
        report_path = self.output_dir / "ncu_analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_header())
            f.write(self._generate_summary())
            f.write(self._generate_gpu_utilization_report())
            f.write(self._generate_memory_report())
            f.write(self._generate_occupancy_report())
            f.write(self._generate_bottleneck_report())
            f.write(self._generate_recommendations())
        
        print(f"📋 报告已生成: {report_path}")
        
        # 同时生成JSON格式的详细数据
        json_path = self.output_dir / "ncu_analysis_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.analyzer.stats, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 详细数据已保存: {json_path}")
        
        # 写入集成报告（新增）
        try:
            report_text = report_path.read_text(encoding='utf-8')
            self._update_integrated_report(report_text)
        except Exception as ie:
            print(f"⚠️ 集成报告更新失败: {ie}")
    
    def _generate_header(self) -> str:
        """生成报告头部"""
        return f"""
{'='*80}
NVIDIA Nsight Compute (NCU) 性能分析报告
{'='*80}
分析文件: {self.parser.input_file}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
分析kernel数量: {len(self.parser.kernels)}
{'='*80}

"""
    
    def _fmt_pct(self, v) -> str:
        try:
            return f"{float(v):.1f}%"
        except Exception:
            return "N/A"

    def _fmt_num(self, v) -> str:
        try:
            return f"{float(v):.1f}"
        except Exception:
            return "N/A"

    def _generate_summary(self) -> str:
        """生成摘要"""
        gpu_stats = self.analyzer.stats.get('gpu_utilization', {})
        avg = self._fmt_pct(gpu_stats.get('average_sm_efficiency', None))
        low = gpu_stats.get('kernels_below_50_percent', 'N/A')
        return f"""
📊 性能摘要
{'-'*40}
• 分析kernel数量: {len(self.parser.kernels)}
• 平均SM效率: {avg} (如果有数据)
• 效率低于50%的kernel数: {low}
• 识别的性能瓶颈: {len(self.analyzer.bottlenecks)}

"""

    def _generate_gpu_utilization_report(self) -> str:
        """生成GPU利用率报告"""
        stats = self.analyzer.stats.get('gpu_utilization', {})
        if 'message' in stats:
            return f"""
🔥 GPU 利用率分析
{'-'*40}
{stats['message']}

"""
        return f"""
🔥 GPU 利用率分析
{'-'*40}
• 平均SM效率: {self._fmt_pct(stats.get('average_sm_efficiency'))}
• 最高SM效率: {self._fmt_pct(stats.get('max_sm_efficiency'))}
• 最低SM效率: {self._fmt_pct(stats.get('min_sm_efficiency'))}
• 效率低于50%的kernel: {stats.get('kernels_below_50_percent', 0)} / {stats.get('total_kernels', 0)}

"""
    
    def _generate_memory_report(self) -> str:
        """生成内存性能报告"""
        stats = self.analyzer.stats.get('memory_analysis', {})
        
        result = f"""
💾 内存性能分析
{'-'*40}
"""
        
        if 'bandwidth_stats' in stats:
            bandwidth = stats['bandwidth_stats']
            result += f"• 平均DRAM带宽: {bandwidth.get('average_bandwidth', 0):.1f} GB/s\n"
            result += f"• 最大DRAM带宽: {bandwidth.get('max_bandwidth', 0):.1f} GB/s\n"
        
        if 'l2_cache_stats' in stats:
            l2_stats = stats['l2_cache_stats']
            result += f"• 平均L2命中率: {l2_stats.get('average_l2_hit_rate', 0):.1f}%\n"
            result += f"• L2命中率低的kernel: {l2_stats.get('kernels_low_l2_hit_rate', 0)}\n"
        
        if 'l1_cache_stats' in stats:
            l1_stats = stats['l1_cache_stats']
            result += f"• 平均L1命中率: {l1_stats.get('average_l1_hit_rate', 0):.1f}%\n"
        
        return result + "\n"
    
    def _generate_occupancy_report(self) -> str:
        """生成占用率报告"""
        stats = self.analyzer.stats.get('occupancy_analysis', {})
        
        result = f"""
🎯 占用率分析
{'-'*40}
"""
        
        if 'achieved_occupancy' in stats:
            achieved = stats['achieved_occupancy']
            result += f"• 平均实际占用率: {achieved.get('average', 0):.1f}%\n"
            result += f"• 占用率范围: {achieved.get('min', 0):.1f}% - {achieved.get('max', 0):.1f}%\n"
        
        if 'occupancy_efficiency' in stats:
            efficiency = stats['occupancy_efficiency']
            result += f"• 平均占用率效率: {efficiency.get('average_ratio', 0)*100:.1f}%\n"
            result += f"• 效率低于80%的kernel: {efficiency.get('kernels_low_efficiency', 0)}\n"
        
        return result + "\n"
    
    def _generate_bottleneck_report(self) -> str:
        """生成瓶颈分析报告"""
        stats = self.analyzer.stats.get('bottleneck_analysis', {})
        
        result = f"""
🚫 性能瓶颈分析
{'-'*40}
• 总瓶颈数量: {stats.get('total_bottlenecks', 0)}
"""
        
        # 瓶颈类型分布
        if 'bottleneck_types' in stats:
            result += "• 瓶颈类型分布:\n"
            for btype, count in stats['bottleneck_types'].items():
                result += f"  - {btype}: {count}\n"
        
        # 主要问题
        if 'top_issues' in stats and stats['top_issues']:
            result += "\n主要性能问题:\n"
            for i, issue in enumerate(stats['top_issues'][:3], 1):
                result += f"{i}. {issue['description']} ({issue['severity']})\n"
                for rec in issue['recommendations'][:2]:
                    result += f"   建议: {rec}\n"
        
        return result + "\n"
    
    def _generate_recommendations(self) -> str:
        """生成优化建议"""
        recommendations = []
        
        # 基于分析结果生成建议
        gpu_stats = self.analyzer.stats.get('gpu_utilization', {})
        if gpu_stats.get('kernels_below_50_percent', 0) > 0:
            recommendations.append("• 有kernel的SM效率低于50%，考虑增加工作负载或优化算法")
        
        memory_stats = self.analyzer.stats.get('memory_analysis', {})
        if 'l2_cache_stats' in memory_stats:
            l2_stats = memory_stats['l2_cache_stats']
            if l2_stats.get('kernels_low_l2_hit_rate', 0) > 0:
                recommendations.append("• 检测到L2缓存命中率低的kernel，优化数据访问模式")
        
        occupancy_stats = self.analyzer.stats.get('occupancy_analysis', {})
        if 'occupancy_efficiency' in occupancy_stats:
            if occupancy_stats['occupancy_efficiency'].get('kernels_low_efficiency', 0) > 0:
                recommendations.append("• 有kernel占用率效率低，检查资源限制(寄存器/共享内存)")
        
        # 默认建议
        if not recommendations:
            recommendations = [
                "• 监控kernel性能指标，识别优化机会",
                "• 考虑使用Tensor Core加速适合的工作负载", 
                "• 优化内存访问模式以提高带宽利用率",
                "• 平衡占用率和每个线程的资源使用"
            ]
        
        return f"""
💡 优化建议
{'-'*40}
{chr(10).join(recommendations)}

{'='*80}
"""
    def _update_integrated_report(self, ncu_text: str):
        """把NCU分析结果插入集成报告，用标记包裹便于后续覆盖"""
        start_tag = "<!-- NCU_REPORT_START -->"
        end_tag   = "<!-- NCU_REPORT_END -->"
        block = f"{start_tag}\n\n{ncu_text}\n{end_tag}\n"

        if INTEGRATED_MD.exists():
            content = INTEGRATED_MD.read_text(encoding='utf-8')
            if start_tag in content and end_tag in content:
                # 替换旧块
                import re
                content = re.sub(f"{start_tag}.*?{end_tag}", block, content, flags=re.DOTALL)
            else:
                # 追加到末尾
                content += ("\n\n" + block)
        else:
            # 初次创建
            header = "# 集成性能分析报告\n\n"
            content = header + block

        INTEGRATED_MD.write_text(content, encoding='utf-8')
        print(f"🧷 已更新集成报告: {INTEGRATED_MD}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='NVIDIA Nsight Compute (NCU) 输出文件自动化解析工具')
    parser.add_argument('input_file', help='输入文件路径 (.ncu-rep, .csv, .json)')
    parser.add_argument('--no-viz', action='store_true', help='不生成可视化图表')
    parser.add_argument('--no-report', action='store_true', help='不生成分析报告')
    parser.add_argument('--output-dir', default='ncu_analysis_output', help='输出目录')
    
    args = parser.parse_args()
    
    try:
        # 解析文件
        print(f"🚀 开始解析NCU文件: {args.input_file}")
        ncu_parser = NCUParser(args.input_file)
        ncu_parser.parse()
        
        # 分析数据
        analyzer = NCUAnalyzer(ncu_parser)
        analyzer.analyze()
        
        # 生成可视化
        if not args.no_viz:
            visualizer = NCUVisualizer(ncu_parser, analyzer)
            visualizer.output_dir = Path(args.output_dir)
            visualizer.output_dir.mkdir(exist_ok=True)
            visualizer.create_visualizations()
        
        # 生成报告
        if not args.no_report:
            reporter = NCUReporter(ncu_parser, analyzer)
            reporter.output_dir = Path(args.output_dir)
            reporter.output_dir.mkdir(exist_ok=True)
            reporter.generate_report()
        
        print(f"\n✅ NCU分析完成! 结果保存在: {args.output_dir}")
        print(f"📊 解析了 {len(ncu_parser.kernels)} 个kernels")
        print(f"🚫 识别了 {len(analyzer.bottlenecks)} 个性能瓶颈")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
