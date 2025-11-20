#!/bin/bash
# Export README.md to PDF with embedded images

cd "$(dirname "$0")"

# Check if pandoc is available
if ! command -v pandoc &> /dev/null; then
    echo "Loading pandoc module..."
    module load pandoc 2>/dev/null || {
        echo "Error: pandoc not found. Please install with:"
        echo "  - On HPC: module load pandoc"
        echo "  - On local: conda install pandoc"
        exit 1
    }
fi

# Convert to PDF
echo "Converting README.md to PDF..."
pandoc README.md \
    -o README.pdf \
    --pdf-engine=xelatex \
    --metadata title="Microstructure Prediction Training Results" \
    --metadata author="LASERNet Project" \
    --metadata date="$(date '+%B %d, %Y')" \
    --toc \
    --toc-depth=2 \
    -V geometry:margin=1in \
    -V linkcolor:blue \
    -V urlcolor:blue \
    --highlight-style=tango

if [ $? -eq 0 ]; then
    echo "✓ Successfully exported to README.pdf"
    ls -lh README.pdf
else
    echo "✗ Export failed"
    exit 1
fi
