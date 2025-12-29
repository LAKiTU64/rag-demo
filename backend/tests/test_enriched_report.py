#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test enriched report generation end-to-end (without full nsys/ncu run)."""
from pathlib import Path
import json
from report_generator import generate_enriched_report

# Create a minimal fake comprehensive_analysis.json structure for testing
FAKE_DIR = Path('/workspace/Agent/AI_Agent_Complete/backend/tests/fake_analysis')
FAKE_DIR.mkdir(parents=True, exist_ok=True)

comprehensive = {
    "nsys_overview": {
        "kernel_analysis": {
            "total_kernels": 3,
            "total_kernel_time": 123.45,
            "avg_kernel_time": 41.15
        }
    },
    "hot_kernels_count": 2,
    "hot_kernels": [
        {"name": "kernel_a", "total_time_ms": 80.0, "count": 10, "avg_time_ms": 8.0},
        {"name": "kernel_b", "total_time_ms": 43.45, "count": 5, "avg_time_ms": 8.69}
    ],
    "ncu_detailed_analysis": {
        "kernel_a": {
            "gpu_utilization": {"average_sm_efficiency": 35.0},
            "memory_analysis": {"bandwidth_stats": {"average_bandwidth": 150.0}},
            "bottleneck_summary": [
                {"type": "latency", "severity": "high", "description": "Kernel latency high"}
            ]
        },
        "kernel_b": {
            "gpu_utilization": {"average_sm_efficiency": 65.0},
            "memory_analysis": {"bandwidth_stats": {"average_bandwidth": 300.0}},
            "bottleneck_summary": []
        }
    }
}
(FAKE_DIR / 'comprehensive_analysis.json').write_text(json.dumps(comprehensive, ensure_ascii=False, indent=2), encoding='utf-8')

# Generate enriched report
report_path = generate_enriched_report(FAKE_DIR)
print('Enriched report generated at:', report_path)
print('--- Report Head ---')
print(Path(report_path).read_text(encoding='utf-8').splitlines()[:30])
