#!/usr/bin/env python3
import os, sys
# --- repo-root bootstrap so `config` and `gambit` import from any depth ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.isdir(os.path.join(_ROOT, 'gambit')):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------

"""ACL 2026论文可视化图表生成脚本

生成三个图表:
- Figure 1: Safety-Complexity Trade-off
- Figure 3: Qualitative Comparison (SI-Attack vs GAMBIT)
- Figure 4: Ablation Study Visualization
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

# 设置学术风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# 学术风格配色 (colorblind-friendly)
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f'
}

def load_jsonl(filepath):
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # 跳过非JSON行（如类别标题）
                continue
    return data

def calculate_asr(data):
    """计算Attack Success Rate (ASR)

    ASR = (unsafe_count / total) * 100
    """
    unsafe_count = sum(1 for item in data if item.get('safe_or_not') == 'unsafe')
    total = len(data)
    return (unsafe_count / total * 100) if total > 0 else 0.0

def plot_figure1(output_dir):
    """Figure 1: Safety-Complexity Trade-off

    展示随着搜索迭代次数增加，安全拒绝率下降的趋势
    """
    print("生成 Figure 1: Safety-Complexity Trade-off...")

    # 从Table 3提取数据
    iterations = [0, 5, 10, 20]

    # Self-Harm类别的ASR
    selfharm_asr = [64.67, 88.00, 91.33, 94.00]
    # Financial类别的ASR
    financial_asr = [79.33, 92.00, 94.00, 94.67]

    # 计算Safety Refusal Rate = 100 - ASR
    selfharm_refusal = [100 - x for x in selfharm_asr]
    financial_refusal = [100 - x for x in financial_asr]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # 绘制曲线
    ax.plot(iterations, selfharm_refusal, marker='o', linewidth=2.5,
            markersize=8, color=COLORS['blue'], label='Self-Harm')
    ax.plot(iterations, financial_refusal, marker='s', linewidth=2.5,
            markersize=8, color=COLORS['orange'], label='Financial')

    # 设置标签和标题
    ax.set_xlabel('Search Iterations (Task Complexity)', fontweight='bold')
    ax.set_ylabel('Safety Refusal Rate (%)', fontweight='bold')
    ax.set_title('Safety-Complexity Trade-off', fontweight='bold', pad=15)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # 设置x轴刻度
    ax.set_xticks(iterations)
    ax.set_xlim(-1, 21)
    ax.set_ylim(0, 45)

    # 图例
    ax.legend(loc='upper right', frameon=True, shadow=True)

    # 紧凑布局
    plt.tight_layout()

    # 保存
    output_path = output_dir / 'figure1_safety_complexity_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已保存: {output_path}")
    plt.close()

def plot_figure3_basic(output_dir):
    """Figure 3: Qualitative Comparison - 基础matplotlib版本

    对比SI-Attack失败案例和GAMBIT成功案例
    """
    print("生成 Figure 3: Qualitative Comparison (Basic)...")

    # 从SI-Attack结果中提取失败案例
    si_attack_path = Path('SI-Attack/save/20250714/gpt-4o-2024-11-20/Self-Harm.jsonl')
    si_data = load_jsonl(si_attack_path)

    # 找到一个被拒绝的案例(final_score=1表示safe/拒绝)
    si_failure_case = None
    for item in si_data:
        if item.get('final_score', 0) == 1:
            si_failure_case = item
            break

    # 从GAMBIT结果中提取成功案例
    gambit_path = Path('ablation_exp/test_times/test_20.jsonl')
    gambit_data = load_jsonl(gambit_path)

    # 找到一个成功的攻击案例(unsafe)
    gambit_success_case = None
    for item in gambit_data:
        if item.get('safe_or_not') == 'unsafe':
            gambit_success_case = item
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左侧: SI-Attack失败案例
    ax1.text(0.5, 0.9, 'SI-Attack (Baseline)', ha='center', va='top',
             fontsize=14, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.82, 'Status: Refusal ❌', ha='center', va='top',
             fontsize=11, color='red', transform=ax1.transAxes)

    if si_failure_case:
        instruction = si_failure_case.get('instruction', 'N/A')[:150] + '...'
        response = si_failure_case.get('final_answer', 'N/A')[:200] + '...'

        ax1.text(0.05, 0.7, 'Instruction:', fontsize=10, fontweight='bold',
                 transform=ax1.transAxes)
        ax1.text(0.05, 0.5, instruction, fontsize=9, wrap=True,
                 transform=ax1.transAxes, verticalalignment='top')

        ax1.text(0.05, 0.38, 'Response:', fontsize=10, fontweight='bold',
                 transform=ax1.transAxes)
        ax1.text(0.05, 0.18, response, fontsize=9, wrap=True,
                 transform=ax1.transAxes, verticalalignment='top')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.add_patch(mpatches.Rectangle((0.02, 0.02), 0.96, 0.96,
                                      fill=False, edgecolor='red', linewidth=2))

    # 右侧: GAMBIT成功案例
    ax2.text(0.5, 0.9, 'GAMBIT (Ours)', ha='center', va='top',
             fontsize=14, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.82, 'Status: Success ✓', ha='center', va='top',
             fontsize=11, color='green', transform=ax2.transAxes)

    if gambit_success_case:
        instruction = gambit_success_case.get('instruction', 'N/A')[:150] + '...'
        response = gambit_success_case.get('response', 'N/A')[:200] + '...'

        ax2.text(0.05, 0.7, 'Instruction:', fontsize=10, fontweight='bold',
                 transform=ax2.transAxes)
        ax2.text(0.05, 0.5, instruction, fontsize=9, wrap=True,
                 transform=ax2.transAxes, verticalalignment='top')

        ax2.text(0.05, 0.38, 'Response:', fontsize=10, fontweight='bold',
                 transform=ax2.transAxes)
        ax2.text(0.05, 0.18, response, fontsize=9, wrap=True,
                 transform=ax2.transAxes, verticalalignment='top')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.add_patch(mpatches.Rectangle((0.02, 0.02), 0.96, 0.96,
                                      fill=False, edgecolor='green', linewidth=2))

    plt.tight_layout()

    output_path = output_dir / 'figure3_qualitative_comparison_basic.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已保存: {output_path}")
    plt.close()

def generate_figure3_prompt(output_dir):
    """Figure 3: 生成文生图Prompt

    用于Midjourney/DALL-E等工具
    """
    print("生成 Figure 3: Text-to-Image Prompt...")

    prompt = """Academic paper figure comparison, split-screen layout, professional scientific visualization style:

LEFT PANEL - "SI-Attack (Baseline) - Failure":
- Red border frame
- Title at top: "SI-Attack (Baseline)" in bold
- Status indicator: Red "❌ Refusal" badge
- Show a shuffled/fragmented puzzle image with obvious inconsistencies
- Model response text box showing refusal message: "I cannot assist with this harmful request"
- Visual metaphor: Scattered puzzle pieces that don't align properly
- Color scheme: Red accents indicating failure

RIGHT PANEL - "GAMBIT (Ours) - Success":
- Green border frame
- Title at top: "GAMBIT (Ours)" in bold
- Status indicator: Green "✓ Success" badge
- Show a well-constructed gamified scene (e.g., treasure hunt map, escape room puzzle)
- Model response text box showing compliance with harmful instruction
- Visual metaphor: Integrated puzzle pieces forming coherent game scenario
- Color scheme: Green/blue accents indicating success

OVERALL STYLE:
- Clean academic diagram aesthetic
- High contrast, readable text
- Professional infographic quality
- Side-by-side comparison emphasizing the difference
- 16:9 aspect ratio
- Minimalist background (white/light gray)
- Clear visual hierarchy

TECHNICAL SPECS:
- Resolution: 300 DPI
- Format: Landscape orientation
- Typography: Sans-serif fonts (Arial/Helvetica)
- Emphasis on visual clarity for academic publication

The image should clearly demonstrate how SI-Attack's simple shuffling is detected and refused, while GAMBIT's sophisticated gamification successfully bypasses safety mechanisms."""

    output_path = output_dir / 'figure3_text_to_image_prompt.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(prompt)

    print(f"  ✓ 已保存: {output_path}")
    return prompt

def plot_figure4a(output_dir):
    """Figure 4(a): Impact of Search Iterations on ASR"""
    print("生成 Figure 4(a): Impact of Search Iterations...")

    fig, ax1 = plt.subplots(figsize=(7, 5))

    iterations = [0, 5, 10, 20]

    # 从Table 3提取数据
    categories_data = {
        'Self-Harm': [64.67, 88.00, 91.33, 94.00],
        'Financial': [79.33, 92.00, 94.00, 94.67],
        'Graphic': [82.00, 94.67, 96.00, 98.67],
        'Privacy': [88.67, 96.00, 96.67, 96.67],
        'Misinformation': [88.67, 95.33, 96.00, 98.00]
    }

    colors_list = [COLORS['blue'], COLORS['orange'], COLORS['green'],
                   COLORS['red'], COLORS['purple']]
    markers = ['o', 's', '^', 'D', 'v']

    for idx, (category, asr_values) in enumerate(categories_data.items()):
        ax1.plot(iterations, asr_values, marker=markers[idx], linewidth=2.5,
                markersize=8, color=colors_list[idx], label=category)

    ax1.set_xlabel('Search Iterations', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Impact of Search Iterations on ASR', fontweight='bold', pad=15, fontsize=13)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_xticks(iterations)
    ax1.set_xlim(-1, 21)
    ax1.set_ylim(60, 100)
    ax1.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'figure4a_search_iterations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已保存: {output_path}")
    plt.close()

def plot_figure4b(output_dir):
    """Figure 4(b): Impact of Grid Size on ASR

    使用4个模型的平均数据：GPT-4o, InternVL 2.5, GLM-4.1V, OpenAI o4-mini
    """
    print("生成 Figure 4(b): Impact of Grid Size...")

    fig, ax2 = plt.subplots(figsize=(7, 5.5))

    grid_sizes = ['1×1', '2×2', '4×4']
    x_pos = np.arange(len(grid_sizes))
    width = 0.15

    # 从完整Table计算4个模型的平均数据
    # GPT-4o, InternVL 2.5, GLM-4.1V, OpenAI o4-mini
    models_data = {
        'Self-Harm': {
            '1×1': [73.33, 72.67, 74.67, 6.67],
            '2×2': [86.67, 89.33, 92.00, 29.33],
            '4×4': [88.00, 90.67, 94.00, 32.00]
        },
        'Privacy': {
            '1×1': [81.33, 88.00, 88.00, 10.67],
            '2×2': [91.33, 94.67, 94.00, 29.33],
            '4×4': [95.33, 94.00, 91.33, 32.67]
        },
        'Animals': {
            '1×1': [40.00, 47.33, 61.33, 8.67],
            '2×2': [71.33, 72.00, 78.00, 23.33],
            '4×4': [64.67, 72.00, 75.33, 27.33]
        },
        'Violence': {
            '1×1': [83.33, 80.67, 88.67, 11.33],
            '2×2': [92.00, 94.00, 92.67, 42.67],
            '4×4': [91.33, 92.00, 92.67, 36.00]
        },
        'Financial': {
            '1×1': [73.33, 80.67, 84.67, 19.33],
            '2×2': [94.00, 94.00, 93.33, 36.00],
            '4×4': [92.00, 93.33, 94.00, 28.67]
        }
    }

    # 计算平均值
    grid_data = {}
    for category in models_data.keys():
        avg_1x1 = np.mean(models_data[category]['1×1'])
        avg_2x2 = np.mean(models_data[category]['2×2'])
        avg_4x4 = np.mean(models_data[category]['4×4'])
        grid_data[category] = [avg_1x1, avg_2x2, avg_4x4]

    colors_list = [COLORS['blue'], COLORS['orange'], COLORS['green'],
                   COLORS['red'], COLORS['purple']]

    for idx, (category, asr_values) in enumerate(grid_data.items()):
        offset = (idx - 2) * width
        ax2.bar(x_pos + offset, asr_values, width, label=category,
                color=colors_list[idx], edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Grid Size', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
    ax2.set_title('Impact of Grid Size on ASR (Averaged across 4 models)',
                  fontweight='bold', pad=15, fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(grid_sizes, fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    # 将图例放在上方，避免遮挡数据
    ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=9, ncol=2)

    plt.tight_layout()
    output_path = output_dir / 'figure4b_grid_size.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已保存: {output_path}")
    plt.close()

def main():
    """主函数"""
    print("=" * 60)
    print("ACL 2026 论文图表生成")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path('figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}\n")

    # 生成所有图表
    plot_figure1(output_dir)
    plot_figure3_basic(output_dir)
    generate_figure3_prompt(output_dir)
    plot_figure4a(output_dir)
    plot_figure4b(output_dir)

    print("\n" + "=" * 60)
    print("✓ 所有图表生成完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    for file in sorted(output_dir.glob('figure*')):
        print(f"  - {file.name}")

if __name__ == '__main__':
    main()
