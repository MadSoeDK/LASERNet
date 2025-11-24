"""
Visualize the architecture of microstructure prediction models.

This script creates detailed flowcharts for both CNN-LSTM and PredRNN models,
showing data flow, layer transformations, and tensor shapes.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


class ModelArchitectureVisualizer:
    """Create detailed architecture flowcharts for neural network models."""

    def __init__(self, figsize=(16, 12)):
        self.figsize = figsize

        # Color scheme
        self.colors = {
            'input': '#E8F4F8',
            'encoder': '#B3E5FC',
            'temporal': '#81D4FA',
            'fusion': '#4FC3F7',
            'decoder': '#29B6F6',
            'output': '#039BE5',
            'pool': '#FFE082',
            'upsample': '#AED581',
        }

    def draw_block(self, ax, x, y, width, height, text, color, fontsize=10, bold=False):
        """Draw a rectangular block with text."""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor=color,
            linewidth=2
        )
        ax.add_patch(box)

        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, weight=weight, wrap=True)

    def draw_arrow(self, ax, x1, y1, x2, y2, label='', style='->'):
        """Draw an arrow between two points."""
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            color='black',
            linewidth=2,
            mutation_scale=20
        )
        ax.add_patch(arrow)

        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    def visualize_cnn_lstm(self, save_path='flowchart_cnn_lstm.png'):
        """Create flowchart for CNN-LSTM model."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')

        # Title
        ax.text(5, 13.5, 'CNN-LSTM Microstructure Prediction Model',
               ha='center', fontsize=16, weight='bold')

        y_pos = 12.5

        # ===== INPUTS =====
        self.draw_block(ax, 2, y_pos, 2.5, 0.6,
                       'Context Sequence\n[B, seq_len, 10, H, W]\n(temp + micro)',
                       self.colors['input'], fontsize=9, bold=True)

        self.draw_block(ax, 8, y_pos, 2.5, 0.6,
                       'Future Temperature\n[B, 1, H, W]',
                       self.colors['input'], fontsize=9, bold=True)

        # ===== NORMALIZATION =====
        y_pos -= 1.2
        self.draw_arrow(ax, 2, 12.2, 2, y_pos + 0.3)
        self.draw_arrow(ax, 8, 12.2, 8, y_pos + 0.3)

        self.draw_block(ax, 2, y_pos, 2.2, 0.5,
                       'Normalize Temp\n(300K - 2000K)',
                       self.colors['input'], fontsize=8)

        self.draw_block(ax, 8, y_pos, 2.2, 0.5,
                       'Normalize Temp\n(300K - 2000K)',
                       self.colors['input'], fontsize=8)

        # ===== CONTEXT ENCODER =====
        y_pos -= 1.0
        self.draw_arrow(ax, 2, 11.05, 2, y_pos + 0.3)

        # Encoder Layer 1
        self.draw_block(ax, 2, y_pos, 2.0, 0.5,
                       'Conv 10→16\n[B, 16, H, W]',
                       self.colors['encoder'], fontsize=8)

        y_pos -= 0.7
        self.draw_arrow(ax, 2, 10.05, 2, y_pos + 0.2)
        self.draw_block(ax, 2, y_pos, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # Encoder Layer 2
        y_pos -= 0.7
        self.draw_arrow(ax, 2, 9.15, 2, y_pos + 0.3)
        self.draw_block(ax, 2, y_pos, 2.0, 0.5,
                       'Conv 16→32\n[B, 32, H/2, W/2]',
                       self.colors['encoder'], fontsize=8)

        y_pos -= 0.7
        self.draw_arrow(ax, 2, 8.05, 2, y_pos + 0.2)
        self.draw_block(ax, 2, y_pos, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # Encoder Layer 3
        y_pos -= 0.7
        self.draw_arrow(ax, 2, 7.15, 2, y_pos + 0.3)
        self.draw_block(ax, 2, y_pos, 2.0, 0.5,
                       'Conv 32→64\n[B, 64, H/4, W/4]',
                       self.colors['encoder'], fontsize=8)

        y_pos -= 0.7
        self.draw_arrow(ax, 2, 6.05, 2, y_pos + 0.2)
        self.draw_block(ax, 2, y_pos, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # ===== FUTURE TEMP ENCODER (right side) =====
        y_pos_future = 10.35
        self.draw_arrow(ax, 8, 11.05, 8, y_pos_future + 0.3)

        # Future Encoder Layer 1
        self.draw_block(ax, 8, y_pos_future, 2.0, 0.5,
                       'Conv 1→16\n[B, 16, H, W]',
                       self.colors['encoder'], fontsize=8)

        y_pos_future -= 0.7
        self.draw_arrow(ax, 8, 10.05, 8, y_pos_future + 0.2)
        self.draw_block(ax, 8, y_pos_future, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # Future Encoder Layer 2
        y_pos_future -= 0.7
        self.draw_arrow(ax, 8, 9.15, 8, y_pos_future + 0.3)
        self.draw_block(ax, 8, y_pos_future, 2.0, 0.5,
                       'Conv 16→32\n[B, 32, H/2, W/2]',
                       self.colors['encoder'], fontsize=8)

        y_pos_future -= 0.7
        self.draw_arrow(ax, 8, 8.05, 8, y_pos_future + 0.2)
        self.draw_block(ax, 8, y_pos_future, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # Future Encoder Layer 3
        y_pos_future -= 0.7
        self.draw_arrow(ax, 8, 7.15, 8, y_pos_future + 0.3)
        self.draw_block(ax, 8, y_pos_future, 2.0, 0.5,
                       'Conv 32→64\n[B, 64, H/4, W/4]',
                       self.colors['encoder'], fontsize=8)

        y_pos_future -= 0.7
        self.draw_arrow(ax, 8, 6.05, 8, y_pos_future + 0.2)
        self.draw_block(ax, 8, y_pos_future, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # ===== CONVLSTM =====
        y_pos -= 0.8
        self.draw_arrow(ax, 2, 5.35, 2, y_pos + 0.4)
        self.draw_block(ax, 2, y_pos, 2.5, 0.7,
                       'ConvLSTM\nTemporal Modeling\n[B, 64, H/8, W/8]',
                       self.colors['temporal'], fontsize=9, bold=True)

        # ===== FUSION =====
        y_pos -= 1.0
        self.draw_arrow(ax, 2, 4.15, 5, y_pos + 0.35)
        self.draw_arrow(ax, 8, 4.85, 5, y_pos + 0.35)

        self.draw_block(ax, 5, y_pos, 2.5, 0.6,
                       'Concatenate\n[B, 128, H/8, W/8]',
                       self.colors['fusion'], fontsize=9, bold=True)

        # ===== DECODER =====
        y_pos -= 0.9
        self.draw_arrow(ax, 5, 2.85, 5, y_pos + 0.2)
        self.draw_block(ax, 5, y_pos, 1.8, 0.4,
                       'Upsample ×2',
                       self.colors['upsample'], fontsize=8)

        y_pos -= 0.6
        self.draw_arrow(ax, 5, 1.75, 5, y_pos + 0.25)
        self.draw_block(ax, 5, y_pos, 2.2, 0.5,
                       'Conv 128→64\n[B, 64, H/4, W/4]',
                       self.colors['decoder'], fontsize=8)

        y_pos -= 0.7
        self.draw_arrow(ax, 5, 0.85, 5, y_pos + 0.2)
        self.draw_block(ax, 5, y_pos, 1.8, 0.4,
                       'Upsample ×2',
                       self.colors['upsample'], fontsize=8)

        y_pos -= 0.6
        self.draw_arrow(ax, 5, 0.05, 5, y_pos + 0.25)
        self.draw_block(ax, 5, y_pos, 2.2, 0.5,
                       'Conv 64→32\n[B, 32, H/2, W/2]',
                       self.colors['decoder'], fontsize=8)

        # Continue decoder on new position
        y_pos = -0.5
        self.draw_arrow(ax, 5, -0.45, 5, y_pos + 0.2)
        self.draw_block(ax, 5, y_pos, 1.8, 0.4,
                       'Upsample ×2',
                       self.colors['upsample'], fontsize=8)

        y_pos -= 0.6
        self.draw_arrow(ax, 5, -0.9, 5, y_pos + 0.25)
        self.draw_block(ax, 5, y_pos, 2.2, 0.5,
                       'Conv 32→16\n[B, 16, H, W]',
                       self.colors['decoder'], fontsize=8)

        # ===== OUTPUT =====
        y_pos -= 0.8
        self.draw_arrow(ax, 5, -1.35, 5, y_pos + 0.3)
        self.draw_block(ax, 5, y_pos, 2.5, 0.6,
                       'Final Conv 16→9\nOutput Microstructure\n[B, 9, H, W]',
                       self.colors['output'], fontsize=9, bold=True)

        # Add legend
        legend_y = -2.8
        ax.text(5, legend_y, 'Legend:', ha='center', fontsize=10, weight='bold')
        legend_items = [
            ('Input/Output', self.colors['input']),
            ('Encoder', self.colors['encoder']),
            ('Temporal (ConvLSTM)', self.colors['temporal']),
            ('Fusion', self.colors['fusion']),
            ('Decoder', self.colors['decoder']),
        ]

        x_legend = 1.5
        for i, (label, color) in enumerate(legend_items):
            self.draw_block(ax, x_legend + i*1.7, legend_y - 0.4, 1.4, 0.3, label, color, fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved CNN-LSTM flowchart to {save_path}")
        plt.close()

    def visualize_predrnn(self, save_path='flowchart_predrnn.png'):
        """Create flowchart for PredRNN model."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')

        # Title
        ax.text(5, 13.5, 'PredRNN Microstructure Prediction Model',
               ha='center', fontsize=16, weight='bold')

        y_pos = 12.5

        # ===== INPUTS =====
        self.draw_block(ax, 2, y_pos, 2.5, 0.6,
                       'Context Sequence\n[B, seq_len, 10, H, W]\n(temp + micro)',
                       self.colors['input'], fontsize=9, bold=True)

        self.draw_block(ax, 8, y_pos, 2.5, 0.6,
                       'Future Temperature\n[B, 1, H, W]',
                       self.colors['input'], fontsize=9, bold=True)

        # ===== NORMALIZATION =====
        y_pos -= 1.2
        self.draw_arrow(ax, 2, 12.2, 2, y_pos + 0.3)
        self.draw_arrow(ax, 8, 12.2, 8, y_pos + 0.3)

        self.draw_block(ax, 2, y_pos, 2.2, 0.5,
                       'Normalize Temp\n(300K - 2000K)',
                       self.colors['input'], fontsize=8)

        self.draw_block(ax, 8, y_pos, 2.2, 0.5,
                       'Normalize Temp\n(300K - 2000K)',
                       self.colors['input'], fontsize=8)

        # ===== CONTEXT ENCODER =====
        y_pos -= 1.0
        self.draw_arrow(ax, 2, 11.05, 2, y_pos + 0.3)

        # Encoder Layer 1
        self.draw_block(ax, 2, y_pos, 2.0, 0.5,
                       'Conv 10→16\n[B, 16, H, W]',
                       self.colors['encoder'], fontsize=8)

        y_pos -= 0.7
        self.draw_arrow(ax, 2, 10.05, 2, y_pos + 0.2)
        self.draw_block(ax, 2, y_pos, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # Encoder Layer 2
        y_pos -= 0.7
        self.draw_arrow(ax, 2, 9.15, 2, y_pos + 0.3)
        self.draw_block(ax, 2, y_pos, 2.0, 0.5,
                       'Conv 16→32\n[B, 32, H/2, W/2]',
                       self.colors['encoder'], fontsize=8)

        y_pos -= 0.7
        self.draw_arrow(ax, 2, 8.05, 2, y_pos + 0.2)
        self.draw_block(ax, 2, y_pos, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # Encoder Layer 3
        y_pos -= 0.7
        self.draw_arrow(ax, 2, 7.15, 2, y_pos + 0.3)
        self.draw_block(ax, 2, y_pos, 2.0, 0.5,
                       'Conv 32→64\n[B, 64, H/4, W/4]',
                       self.colors['encoder'], fontsize=8)

        y_pos -= 0.7
        self.draw_arrow(ax, 2, 6.05, 2, y_pos + 0.2)
        self.draw_block(ax, 2, y_pos, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # ===== FUTURE TEMP ENCODER (right side) =====
        y_pos_future = 10.35
        self.draw_arrow(ax, 8, 11.05, 8, y_pos_future + 0.3)

        # Future Encoder Layer 1
        self.draw_block(ax, 8, y_pos_future, 2.0, 0.5,
                       'Conv 1→16\n[B, 16, H, W]',
                       self.colors['encoder'], fontsize=8)

        y_pos_future -= 0.7
        self.draw_arrow(ax, 8, 10.05, 8, y_pos_future + 0.2)
        self.draw_block(ax, 8, y_pos_future, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # Future Encoder Layer 2
        y_pos_future -= 0.7
        self.draw_arrow(ax, 8, 9.15, 8, y_pos_future + 0.3)
        self.draw_block(ax, 8, y_pos_future, 2.0, 0.5,
                       'Conv 16→32\n[B, 32, H/2, W/2]',
                       self.colors['encoder'], fontsize=8)

        y_pos_future -= 0.7
        self.draw_arrow(ax, 8, 8.05, 8, y_pos_future + 0.2)
        self.draw_block(ax, 8, y_pos_future, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # Future Encoder Layer 3
        y_pos_future -= 0.7
        self.draw_arrow(ax, 8, 7.15, 8, y_pos_future + 0.3)
        self.draw_block(ax, 8, y_pos_future, 2.0, 0.5,
                       'Conv 32→64\n[B, 64, H/4, W/4]',
                       self.colors['encoder'], fontsize=8)

        y_pos_future -= 0.7
        self.draw_arrow(ax, 8, 6.05, 8, y_pos_future + 0.2)
        self.draw_block(ax, 8, y_pos_future, 1.5, 0.4,
                       'MaxPool ÷2',
                       self.colors['pool'], fontsize=8)

        # ===== PREDRNN =====
        y_pos -= 0.8
        self.draw_arrow(ax, 2, 5.35, 2, y_pos + 0.5)
        self.draw_block(ax, 2, y_pos, 2.8, 0.9,
                       'PredRNN\nST-LSTM Layers (4)\nSpatiotemporal\nModeling\n[B, 64, H/8, W/8]',
                       self.colors['temporal'], fontsize=9, bold=True)

        # ===== FUSION =====
        y_pos -= 1.2
        self.draw_arrow(ax, 2, 3.65, 5, y_pos + 0.35)
        self.draw_arrow(ax, 8, 4.85, 5, y_pos + 0.35)

        self.draw_block(ax, 5, y_pos, 2.5, 0.6,
                       'Concatenate\n[B, 128, H/8, W/8]',
                       self.colors['fusion'], fontsize=9, bold=True)

        # ===== DECODER =====
        y_pos -= 0.9
        self.draw_arrow(ax, 5, 2.15, 5, y_pos + 0.2)
        self.draw_block(ax, 5, y_pos, 1.8, 0.4,
                       'Upsample ×2',
                       self.colors['upsample'], fontsize=8)

        y_pos -= 0.6
        self.draw_arrow(ax, 5, 1.05, 5, y_pos + 0.25)
        self.draw_block(ax, 5, y_pos, 2.2, 0.5,
                       'Conv 128→64\n[B, 64, H/4, W/4]',
                       self.colors['decoder'], fontsize=8)

        y_pos -= 0.7
        self.draw_arrow(ax, 5, 0.15, 5, y_pos + 0.2)
        self.draw_block(ax, 5, y_pos, 1.8, 0.4,
                       'Upsample ×2',
                       self.colors['upsample'], fontsize=8)

        y_pos -= 0.6
        self.draw_arrow(ax, 5, -0.65, 5, y_pos + 0.25)
        self.draw_block(ax, 5, y_pos, 2.2, 0.5,
                       'Conv 64→32\n[B, 32, H/2, W/2]',
                       self.colors['decoder'], fontsize=8)

        # Continue decoder
        y_pos -= 0.7
        self.draw_arrow(ax, 5, -1.35, 5, y_pos + 0.2)
        self.draw_block(ax, 5, y_pos, 1.8, 0.4,
                       'Upsample ×2',
                       self.colors['upsample'], fontsize=8)

        y_pos -= 0.6
        self.draw_arrow(ax, 5, -2.15, 5, y_pos + 0.25)
        self.draw_block(ax, 5, y_pos, 2.2, 0.5,
                       'Conv 32→16\n[B, 16, H, W]',
                       self.colors['decoder'], fontsize=8)

        # ===== OUTPUT =====
        y_pos -= 0.8
        self.draw_arrow(ax, 5, -3.05, 5, y_pos + 0.3)
        self.draw_block(ax, 5, y_pos, 2.5, 0.6,
                       'Final Conv 16→9\nOutput Microstructure\n[B, 9, H, W]',
                       self.colors['output'], fontsize=9, bold=True)

        # Add legend
        legend_y = -4.5
        ax.text(5, legend_y, 'Legend:', ha='center', fontsize=10, weight='bold')
        legend_items = [
            ('Input/Output', self.colors['input']),
            ('Encoder', self.colors['encoder']),
            ('Temporal (PredRNN)', self.colors['temporal']),
            ('Fusion', self.colors['fusion']),
            ('Decoder', self.colors['decoder']),
        ]

        x_legend = 1.5
        for i, (label, color) in enumerate(legend_items):
            self.draw_block(ax, x_legend + i*1.7, legend_y - 0.4, 1.4, 0.3, label, color, fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved PredRNN flowchart to {save_path}")
        plt.close()


def main():
    """Generate flowcharts for both models."""
    print("=" * 70)
    print("Microstructure Model Architecture Visualization")
    print("=" * 70)
    print()

    visualizer = ModelArchitectureVisualizer(figsize=(16, 14))

    print("Generating CNN-LSTM flowchart...")
    visualizer.visualize_cnn_lstm('flowchart_cnn_lstm.png')
    print()

    print("Generating PredRNN flowchart...")
    visualizer.visualize_predrnn('flowchart_predrnn.png')
    print()

    print("=" * 70)
    print("Done! Generated flowcharts:")
    print("  1. flowchart_cnn_lstm.png")
    print("  2. flowchart_predrnn.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
