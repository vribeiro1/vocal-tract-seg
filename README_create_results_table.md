# Results Table Generator

🎯 A Python tool to generate statistical tables from Mask R-CNN test results for vocal tract segmentation.

## Table of Contents
- [What It Does](#what-it-does)
- [Quick Start](#quick-start)
- [Output Formats](#output-formats)
- [Common Use Cases](#common-use-cases)
- [How It Works](#how-it-works)
- [Command Reference](#command-reference)
---

## What It Does

This script analyzes your `test_results.csv` files and automatically generates formatted tables with statistics for each articulator (tongue, lips, pharynx, etc.). It's smart enough to detect whether you have one subject or multiple subjects and adapts accordingly.

## Quick Start

### For Single Subject (Auto Mode)
Just run with your CSV file - it automatically generates statistics:

```bash
python create_results_table.py vocal-tract-seg/results/32/test_results.csv
```

**Output:**
```
STATISTICS: Subject 1640
====================================================================================================
Articulator               P2CP_RMS (mm)                       Jaccard Index                      
                          Mean±Std             Range           Mean±Std             Range          
----------------------------------------------------------------------------------------------------
Arytenoid Cartilage       1.07±0.84            [0.48, 4.54]    -                    -              
Epiglottis                1.22±0.97            [0.33, 4.94]    -                    -              
Lower Lip                 0.58±0.14            [0.40, 0.95]    -                    -              
...
```

### For Multiple Subjects (Comparison Mode)
Compare two subjects or sequences:

```bash
# Compare two subjects (aggregates all their sequences)
python create_results_table.py merged_subjects.csv --compare subject -s1 1618 -s2 1640

# Compare two sequences
python create_results_table.py results.csv --compare sequence -s1 S10 -s2 S14
```

**Output includes:**
- Mean ± Std for both items
- Statistical p-values (t-tests)
- Sample counts

## Output Formats

Choose your output format with `-f`:

```bash
# Text format (default) - for terminal viewing
python create_results_table.py results.csv

# LaTeX format - for papers and thesis
python create_results_table.py results.csv -f latex

# Markdown format - for GitHub README or docs
python create_results_table.py results.csv -f markdown
```

## Save to File

Use `-o` to save output directly:

```bash
# Save LaTeX table for your paper
python create_results_table.py results.csv -f latex -o paper_table.tex

# Save markdown for documentation
python create_results_table.py results.csv -f markdown -o results.md
```

## Common Use Cases

### Use Case 1: Analyze Single Experiment Results
```bash
# One command - shows statistics automatically
python create_results_table.py vocal-tract-seg/results/32/test_results.csv
```

**When to use:** You've just finished training and want to see how well your model performs.

---

### Use Case 2: Compare Different Sequences
```bash
# Compare performance on S10 vs S14 sequences
python create_results_table.py vocal-tract-seg/results/32/test_results.csv \
    --compare sequence -s1 S10 -s2 S14
```

**When to use:** Check if your model performs differently on various speech sequences.

---

### Use Case 3: Compare Two Subjects
```bash
# First, merge results from different experiments
python merge_results.py

# Then compare subjects
python create_results_table.py merged_subjects.csv \
    --compare subject -s1 1618 -s2 1640
```

**When to use:** Compare model performance across different subjects/speakers.

---

### Use Case 4: Generate Tables for Thesis/Paper
```bash
# Generate LaTeX table for your thesis
python create_results_table.py vocal-tract-seg/results/32/test_results.csv \
    -f latex -o chapter4_results.tex
```

**When to use:** Writing your thesis or paper and need publication-ready tables.

---

### Use Case 5: Create Documentation
```bash
# Generate markdown table for README
python create_results_table.py vocal-tract-seg/results/32/test_results.csv \
    -f markdown -o README_results.md
```

**When to use:** Documenting your project on GitHub.

## How It Works

```
┌─────────────────────────────┐
│ Load test_results.csv       │
│ - subject, sequence, frame  │
│ - p2cp_rms, jaccard_index   │
└──────────┬──────────────────┘
           │
           ▼
    ┌──────────────┐
    │ Detect mode  │
    └──────┬───────┘
           │
    ┌──────┴──────┐
    ▼             ▼
[1 subject]  [2+ subjects]
    │             │
    ▼             ▼
Auto-gen      Need --compare
statistics    flag to compare
```

## Command Reference

### Basic Syntax
```bash
python create_results_table.py <csv_path> [options]
```

### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--compare`, `-c` | Comparison mode: `subject` or `sequence` | `--compare subject` |
| `--subject1`, `-s1` | First item to compare | `-s1 1618` or `-s1 S10` |
| `--subject2`, `-s2` | Second item to compare | `-s2 1640` or `-s2 S14` |
| `--format`, `-f` | Output format: `text`, `latex`, `markdown` | `-f latex` |
| `--output`, `-o` | Save to file | `-o results.tex` |

## What Gets Analyzed

### Metrics
- **P2CP_RMS**: Point-to-curve-point root mean square distance (mm)
- **Jaccard Index**: Overlap similarity between predicted and ground truth

### Articulators (9 vocal tract structures)
1. Arytenoid Cartilage
2. Epiglottis
3. Lower Lip
4. Pharynx
5. Soft Palate Midline
6. Thyroid Cartilage
7. Tongue
8. Upper Lip
9. Vocal Folds

### Statistics Provided

**For Single Subject:**
- Mean ± Standard Deviation
- Range [Min, Max]
- Total measurements
- Sequences included

**For Comparisons:**
- Mean ± Std for both items
- p-values (statistical significance)
- Sample counts
- Which sequences are in each comparison

## Examples

### Example 1: Quick Check
```bash
# Just see what's in your data
python create_results_table.py vocal-tract-seg/results/32/test_results.csv
```

### Example 2: Detailed Comparison
```bash
# Compare two sequences with LaTeX output
python create_results_table.py vocal-tract-seg/results/32/test_results.csv \
    --compare sequence \
    -s1 S10 \
    -s2 S14 \
    -f latex \
    -o seq_comparison.tex
```

### Example 3: Multi-Experiment Analysis
```bash
# Step 1: Merge CSVs from experiments 25 and 32
python merge_results.py

# Step 2: Compare subjects across experiments
python create_results_table.py merged_subjects.csv \
    --compare subject \
    -s1 1618 \
    -s2 1640 \
    -f latex \
    -o subject_comparison.tex
```