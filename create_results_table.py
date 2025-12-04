#!/usr/bin/env python3
"""
Object-Oriented Results Table Generator
Create formatted results tables from test_results.csv
Computes statistics per articulator comparing two subjects or sequences
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import os
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class StatisticsResult:
    """Data class to store statistical results"""
    mean: tuple
    std: tuple
    pvalue: float = None


class ResultsDataLoader:
    """Class to load and manage test results data"""
    
    def __init__(self, csv_path: str):
        """
        Initialize and load CSV data
        
        Parameters:
        -----------
        csv_path : str
            Path to test_results.csv file
        """
        self.csv_path = csv_path
        self.df = None
        self.subjects = []
        self.sequences = []
        self.articulators = []
        
        self._load_and_analyze()
    
    def _load_and_analyze(self):
        """Load CSV and analyze its contents"""
        try:
            self.df = pd.read_csv(self.csv_path)
            self._extract_metadata()
            self._print_summary()
        except FileNotFoundError:
            print(f"Error: File not found - {self.csv_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            sys.exit(1)
    
    def _extract_metadata(self):
        """Extract metadata from dataframe"""
        self.subjects = sorted(self.df['subject'].unique())
        self.sequences = sorted(self.df['sequence'].unique())
        self.articulators = sorted(self.df['pred_class'].unique())
    
    def _print_summary(self):
        """Print data summary"""
        print(f"\n{'='*80}")
        print(f"📂 LOADED: {self.csv_path}")
        print(f"{'='*80}")
        print(f"Total measurements: {len(self.df)}")
        
        # Subject information
        print(f"\n📊 SUBJECTS ({len(self.subjects)}):")
        for subject in self.subjects:
            subject_data = self.df[self.df['subject'] == subject]
            sequences = sorted(subject_data['sequence'].unique())
            print(f"  Subject {subject}:")
            print(f"    • Sequences: {sequences}")
            print(f"    • Total measurements: {len(subject_data)}")
            for seq in sequences:
                seq_count = len(subject_data[subject_data['sequence'] == seq])
                print(f"      - {seq}: {seq_count} measurements")
        
        # Sequence information
        print(f"\n📋 SEQUENCES ({len(self.sequences)}):")
        for seq in self.sequences:
            seq_data = self.df[self.df['sequence'] == seq]
            subjects_in_seq = sorted(seq_data['subject'].unique())
            print(f"  {seq}: {len(seq_data)} measurements (subjects: {subjects_in_seq})")
        
        # Articulator information
        print(f"\n🔬 ARTICULATORS ({len(self.articulators)}):")
        for art in self.articulators:
            count = len(self.df[self.df['pred_class'] == art])
            print(f"  • {art}: {count} measurements")
        
        # Metrics available
        print(f"\n📈 AVAILABLE METRICS:")
        print(f"  • P2CP_RMS: {self.df['p2cp_rms'].notna().sum()} values")
        print(f"  • P2CP_MEAN: {self.df['p2cp_mean'].notna().sum()} values")
        print(f"  • Jaccard Index: {self.df['jaccard_index'].notna().sum()} values")
        print(f"{'='*80}\n")
    
    def get_data_by_subject(self, subject) -> pd.DataFrame:
        """Get all data for a specific subject (all sequences)"""
        return self.df[self.df['subject'].astype(str) == str(subject)].copy()
    
    def get_data_by_sequence(self, sequence: str) -> pd.DataFrame:
        """Get all data for a specific sequence"""
        return self.df[self.df['sequence'] == sequence].copy()


class ComparisonEngine:
    """Engine to perform statistical comparisons"""
    
    def __init__(self, data_loader: ResultsDataLoader):
        """Initialize with data loader"""
        self.data = data_loader
    
    def compare(self, item1, item2, compare_by: str = 'sequence') -> Dict:
        """
        Compare two items (subjects or sequences)
        
        Parameters:
        -----------
        item1 : str or int
            First item to compare
        item2 : str or int
            Second item to compare
        compare_by : str
            'sequence' or 'subject'
        
        Returns:
        --------
        Dict containing comparison results for P2CP and Jaccard metrics
        """
        # Get data
        if compare_by == 'subject':
            df1 = self.data.get_data_by_subject(item1)
            df2 = self.data.get_data_by_subject(item2)
            label1 = f"Subject {item1}"
            label2 = f"Subject {item2}"
        else:
            df1 = self.data.get_data_by_sequence(item1)
            df2 = self.data.get_data_by_sequence(item2)
            label1 = str(item1)
            label2 = str(item2)
        
        if len(df1) == 0 or len(df2) == 0:
            print(f"Error: No data found for comparison")
            return None
        
        # Perform comparisons
        results = {
            'labels': (label1, label2),
            'items': (item1, item2),
            'compare_by': compare_by,
            'count1': len(df1),
            'count2': len(df2),
            'p2cp': {},
            'jaccard': {},
            'overall_p2cp': None,
            'overall_jaccard': None
        }
        
        # Per articulator comparison
        for articulator in self.data.articulators:
            # P2CP comparison
            art1_p2cp = df1[df1['pred_class'] == articulator]['p2cp_rms'].dropna()
            art2_p2cp = df2[df2['pred_class'] == articulator]['p2cp_rms'].dropna()
            
            if len(art1_p2cp) > 0 and len(art2_p2cp) > 0:
                _, pval = stats.ttest_ind(art1_p2cp, art2_p2cp)
                results['p2cp'][articulator] = StatisticsResult(
                    mean=(art1_p2cp.mean(), art2_p2cp.mean()),
                    std=(art1_p2cp.std(), art2_p2cp.std()),
                    pvalue=pval
                )
            
            # Jaccard comparison
            art1_jacc = df1[df1['pred_class'] == articulator]['jaccard_index'].dropna()
            art2_jacc = df2[df2['pred_class'] == articulator]['jaccard_index'].dropna()
            
            if len(art1_jacc) > 0 and len(art2_jacc) > 0:
                _, pval = stats.ttest_ind(art1_jacc, art2_jacc)
                results['jaccard'][articulator] = StatisticsResult(
                    mean=(art1_jacc.mean(), art2_jacc.mean()),
                    std=(art1_jacc.std(), art2_jacc.std()),
                    pvalue=pval
                )
        
        # Overall statistics
        p2cp1_all = df1['p2cp_rms'].dropna()
        p2cp2_all = df2['p2cp_rms'].dropna()
        if len(p2cp1_all) > 0 and len(p2cp2_all) > 0:
            _, pval = stats.ttest_ind(p2cp1_all, p2cp2_all)
            results['overall_p2cp'] = StatisticsResult(
                mean=(p2cp1_all.mean(), p2cp2_all.mean()),
                std=(p2cp1_all.std(), p2cp2_all.std()),
                pvalue=pval
            )
        
        jacc1_all = df1['jaccard_index'].dropna()
        jacc2_all = df2['jaccard_index'].dropna()
        if len(jacc1_all) > 0 and len(jacc2_all) > 0:
            _, pval = stats.ttest_ind(jacc1_all, jacc2_all)
            results['overall_jaccard'] = StatisticsResult(
                mean=(jacc1_all.mean(), jacc2_all.mean()),
                std=(jacc1_all.std(), jacc2_all.std()),
                pvalue=pval
            )
        
        # Add sequence info if comparing subjects
        if compare_by == 'subject':
            results['seq1'] = sorted(df1['sequence'].unique())
            results['seq2'] = sorted(df2['sequence'].unique())
        
        return results
    
    def generate_single_stats(self, item, compare_by: str = 'subject') -> Dict:
        """
        Generate statistics for a single item (subject or sequence)
        
        Parameters:
        -----------
        item : str or int
            Item to analyze
        compare_by : str
            'sequence' or 'subject'
        
        Returns:
        --------
        Dict containing statistics for P2CP and Jaccard metrics
        """
        # Get data
        if compare_by == 'subject':
            df = self.data.get_data_by_subject(item)
            label = f"Subject {item}"
        else:
            df = self.data.get_data_by_sequence(item)
            label = str(item)
        
        if len(df) == 0:
            print(f"Error: No data found for {label}")
            return None
        
        # Perform analysis
        results = {
            'label': label,
            'item': item,
            'compare_by': compare_by,
            'count': len(df),
            'p2cp': {},
            'jaccard': {},
            'overall_p2cp': None,
            'overall_jaccard': None
        }
        
        # Per articulator statistics
        for articulator in self.data.articulators:
            # P2CP statistics
            art_p2cp = df[df['pred_class'] == articulator]['p2cp_rms'].dropna()
            
            if len(art_p2cp) > 0:
                results['p2cp'][articulator] = {
                    'mean': art_p2cp.mean(),
                    'std': art_p2cp.std(),
                    'min': art_p2cp.min(),
                    'max': art_p2cp.max(),
                    'count': len(art_p2cp)
                }
            
            # Jaccard statistics
            art_jacc = df[df['pred_class'] == articulator]['jaccard_index'].dropna()
            
            if len(art_jacc) > 0:
                results['jaccard'][articulator] = {
                    'mean': art_jacc.mean(),
                    'std': art_jacc.std(),
                    'min': art_jacc.min(),
                    'max': art_jacc.max(),
                    'count': len(art_jacc)
                }
        
        # Overall statistics
        p2cp_all = df['p2cp_rms'].dropna()
        if len(p2cp_all) > 0:
            results['overall_p2cp'] = {
                'mean': p2cp_all.mean(),
                'std': p2cp_all.std(),
                'min': p2cp_all.min(),
                'max': p2cp_all.max(),
                'count': len(p2cp_all)
            }
        
        jacc_all = df['jaccard_index'].dropna()
        if len(jacc_all) > 0:
            results['overall_jaccard'] = {
                'mean': jacc_all.mean(),
                'std': jacc_all.std(),
                'min': jacc_all.min(),
                'max': jacc_all.max(),
                'count': len(jacc_all)
            }
        
        # Add sequence info if analyzing subject
        if compare_by == 'subject':
            results['sequences'] = sorted(df['sequence'].unique())
        
        return results


class TableFormatter:
    """Class to format and print comparison tables"""
    
    @staticmethod
    def format_pvalue(p: float, format_type: str = 'text') -> str:
        """Format p-value based on output type"""
        if pd.isna(p):
            return "" if format_type == 'latex' else "-"
        
        if format_type == 'latex':
            if p < 0.001:
                exponent = int(np.floor(np.log10(p)))
                mantissa = p / (10 ** exponent)
                if mantissa == 1.0:
                    return f"$10^{{{exponent}}}$"
                else:
                    return f"${mantissa:.1f} \\times 10^{{{exponent}}}$"
            return f"{p:.2f}"
        else:
            return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
    
    @staticmethod
    def print_text_table(results: Dict):
        """Print text format table"""
        label1, label2 = results['labels']
        
        print(f"\n{'='*100}")
        print(f"COMPARISON: {label1} vs {label2}")
        print(f"{'='*100}\n")
        
        # Header
        print(f"{'Articulator':<25} {'P2CP_RMS (mm)':<40} {'Jaccard Index':<40}")
        print(f"{'':<25} {label1:<18} {label2:<18} {'p-val':<4} {label1:<18} {label2:<18} {'p-val':<4}")
        print("-" * 110)
        
        # Per articulator
        all_articulators = set(results['p2cp'].keys()) | set(results['jaccard'].keys())
        for art in sorted(all_articulators):
            art_name = art.replace('-', ' ').title()
            
            # P2CP
            if art in results['p2cp']:
                r = results['p2cp'][art]
                p2cp1_str = f"{r.mean[0]:.2f}±{r.std[0]:.2f}"
                p2cp2_str = f"{r.mean[1]:.2f}±{r.std[1]:.2f}"
                p2cp_pval = TableFormatter.format_pvalue(r.pvalue, 'text')
            else:
                p2cp1_str = p2cp2_str = p2cp_pval = "-"
            
            # Jaccard
            if art in results['jaccard']:
                r = results['jaccard'][art]
                jacc1_str = f"{r.mean[0]:.2f}±{r.std[0]:.2f}"
                jacc2_str = f"{r.mean[1]:.2f}±{r.std[1]:.2f}"
                jacc_pval = TableFormatter.format_pvalue(r.pvalue, 'text')
            else:
                jacc1_str = jacc2_str = jacc_pval = "-"
            
            print(f"{art_name:<25} {p2cp1_str:<18} {p2cp2_str:<18} {p2cp_pval:<4} "
                  f"{jacc1_str:<18} {jacc2_str:<18} {jacc_pval:<4}")
        
        # Overall
        print("-" * 110)
        r_p2cp = results['overall_p2cp']
        r_jacc = results['overall_jaccard']
        
        if r_p2cp:
            p2cp1_str = f"{r_p2cp.mean[0]:.2f}±{r_p2cp.std[0]:.2f}"
            p2cp2_str = f"{r_p2cp.mean[1]:.2f}±{r_p2cp.std[1]:.2f}"
            p2cp_pval = TableFormatter.format_pvalue(r_p2cp.pvalue, 'text')
        else:
            p2cp1_str = p2cp2_str = p2cp_pval = "-"
        
        if r_jacc:
            jacc1_str = f"{r_jacc.mean[0]:.2f}±{r_jacc.std[0]:.2f}"
            jacc2_str = f"{r_jacc.mean[1]:.2f}±{r_jacc.std[1]:.2f}"
            jacc_pval = TableFormatter.format_pvalue(r_jacc.pvalue, 'text')
        else:
            jacc1_str = jacc2_str = jacc_pval = "-"
        
        print(f"{'OVERALL':<25} {p2cp1_str:<18} {p2cp2_str:<18} {p2cp_pval:<4} "
              f"{jacc1_str:<18} {jacc2_str:<18} {jacc_pval:<4}")
        
        # Summary
        print(f"\n{'='*100}")
        print(f"Sample counts:")
        print(f"  {label1}: {results['count1']} measurements")
        print(f"  {label2}: {results['count2']} measurements")
        
        if 'seq1' in results:
            print(f"    Sequences in {label1}: {results['seq1']}")
            print(f"    Sequences in {label2}: {results['seq2']}")
        
        print(f"{'='*100}\n")
    
    @staticmethod
    def print_latex_table(results: Dict):
        """Print LaTeX format table"""
        label1, label2 = results['labels']
        
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{lcccccc}")
        print("\\hline")
        print("& \\multicolumn{3}{c}{P2CP$_{RMS}$ (mm)} & \\multicolumn{3}{c}{Jaccard index} \\\\")
        print(f"Articulator & {label1} & {label2} & $p$-value & {label1} & {label2} & $p$-value \\\\")
        print("\\hline")
        
        # Per articulator
        all_articulators = set(results['p2cp'].keys()) | set(results['jaccard'].keys())
        for art in sorted(all_articulators):
            art_name = art.replace('-', ' ').title()
            if 'Midline' in art_name:
                art_name = art_name.replace('Soft Palate Midline', 'Soft Palate\\newline Center line')
            
            # P2CP
            if art in results['p2cp']:
                r = results['p2cp'][art]
                p2cp1_str = f"{r.mean[0]:.2f} $\\pm$ {r.std[0]:.2f}"
                p2cp2_str = f"{r.mean[1]:.2f} $\\pm$ {r.std[1]:.2f}"
                p2cp_pval = TableFormatter.format_pvalue(r.pvalue, 'latex')
            else:
                p2cp1_str = p2cp2_str = p2cp_pval = ""
            
            # Jaccard
            if art in results['jaccard']:
                r = results['jaccard'][art]
                jacc1_str = f"{r.mean[0]:.2f} $\\pm$ {r.std[0]:.2f}"
                jacc2_str = f"{r.mean[1]:.2f} $\\pm$ {r.std[1]:.2f}"
                jacc_pval = TableFormatter.format_pvalue(r.pvalue, 'latex')
            else:
                jacc1_str = jacc2_str = jacc_pval = ""
            
            print(f"{art_name} & {p2cp1_str} & {p2cp2_str} & {p2cp_pval} & "
                  f"{jacc1_str} & {jacc2_str} & {jacc_pval} \\\\")
        
        # Overall
        print("\\hline")
        r_p2cp = results['overall_p2cp']
        r_jacc = results['overall_jaccard']
        
        if r_p2cp:
            p2cp1_str = f"{r_p2cp.mean[0]:.2f} $\\pm$ {r_p2cp.std[0]:.2f}"
            p2cp2_str = f"{r_p2cp.mean[1]:.2f} $\\pm$ {r_p2cp.std[1]:.2f}"
            p2cp_pval = TableFormatter.format_pvalue(r_p2cp.pvalue, 'latex')
        else:
            p2cp1_str = p2cp2_str = p2cp_pval = ""
        
        if r_jacc:
            jacc1_str = f"{r_jacc.mean[0]:.2f} $\\pm$ {r_jacc.std[0]:.2f}"
            jacc2_str = f"{r_jacc.mean[1]:.2f} $\\pm$ {r_jacc.std[1]:.2f}"
            jacc_pval = TableFormatter.format_pvalue(r_jacc.pvalue, 'latex')
        else:
            jacc1_str = jacc2_str = jacc_pval = ""
        
        print(f"mean $\\pm$ std & {p2cp1_str} & {p2cp2_str} & {p2cp_pval} & "
              f"{jacc1_str} & {jacc2_str} & {jacc_pval} \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print("\\caption{Comparison of segmentation performance}")
        print("\\end{table}")
    
    @staticmethod
    def print_markdown_table(results: Dict):
        """Print Markdown format table"""
        label1, label2 = results['labels']
        
        print(f"\n## Comparison: {label1} vs {label2}\n")
        print("| Articulator | P2CP_RMS (mm) | | | Jaccard Index | | |")
        print("|-------------|---------------|---------------|---------|---------------|---------------|---------|")
        print(f"|             | {label1} | {label2} | p-value | {label1} | {label2} | p-value |")
        print("|-------------|---------------|---------------|---------|---------------|---------------|---------|")
        
        # Per articulator
        all_articulators = set(results['p2cp'].keys()) | set(results['jaccard'].keys())
        for art in sorted(all_articulators):
            art_name = art.replace('-', ' ').title()
            
            # P2CP
            if art in results['p2cp']:
                r = results['p2cp'][art]
                p2cp1_str = f"{r.mean[0]:.2f} ± {r.std[0]:.2f}"
                p2cp2_str = f"{r.mean[1]:.2f} ± {r.std[1]:.2f}"
                p2cp_pval = f"{r.pvalue:.4f}" if r.pvalue >= 0.001 else f"{r.pvalue:.2e}"
            else:
                p2cp1_str = p2cp2_str = p2cp_pval = "-"
            
            # Jaccard
            if art in results['jaccard']:
                r = results['jaccard'][art]
                jacc1_str = f"{r.mean[0]:.2f} ± {r.std[0]:.2f}"
                jacc2_str = f"{r.mean[1]:.2f} ± {r.std[1]:.2f}"
                jacc_pval = f"{r.pvalue:.4f}" if r.pvalue >= 0.001 else f"{r.pvalue:.2e}"
            else:
                jacc1_str = jacc2_str = jacc_pval = "-"
            
            print(f"| {art_name} | {p2cp1_str} | {p2cp2_str} | {p2cp_pval} | "
                  f"{jacc1_str} | {jacc2_str} | {jacc_pval} |")
        
        # Overall
        r_p2cp = results['overall_p2cp']
        r_jacc = results['overall_jaccard']
        
        if r_p2cp:
            p2cp1_str = f"{r_p2cp.mean[0]:.2f} ± {r_p2cp.std[0]:.2f}"
            p2cp2_str = f"{r_p2cp.mean[1]:.2f} ± {r_p2cp.std[1]:.2f}"
            p2cp_pval = f"{r_p2cp.pvalue:.4f}" if r_p2cp.pvalue >= 0.001 else f"{r_p2cp.pvalue:.2e}"
        else:
            p2cp1_str = p2cp2_str = p2cp_pval = "-"
        
        if r_jacc:
            jacc1_str = f"{r_jacc.mean[0]:.2f} ± {r_jacc.std[0]:.2f}"
            jacc2_str = f"{r_jacc.mean[1]:.2f} ± {r_jacc.std[1]:.2f}"
            jacc_pval = f"{r_jacc.pvalue:.4f}" if r_jacc.pvalue >= 0.001 else f"{r_jacc.pvalue:.2e}"
        else:
            jacc1_str = jacc2_str = jacc_pval = "-"
        
        print(f"| mean ± std | {p2cp1_str} | {p2cp2_str} | {p2cp_pval} | "
              f"{jacc1_str} | {jacc2_str} | {jacc_pval} |")
    
    @staticmethod
    def print_single_text_table(results: Dict):
        """Print text format table for single subject/sequence"""
        label = results['label']
        
        print(f"\n{'='*100}")
        print(f"STATISTICS: {label}")
        print(f"{'='*100}\n")
        
        # Header
        print(f"{'Articulator':<25} {'P2CP_RMS (mm)':<35} {'Jaccard Index':<35}")
        print(f"{'':<25} {'Mean±Std':<20} {'Range':<15} {'Mean±Std':<20} {'Range':<15}")
        print("-" * 100)
        
        # Per articulator
        all_articulators = set(results['p2cp'].keys()) | set(results['jaccard'].keys())
        for art in sorted(all_articulators):
            art_name = art.replace('-', ' ').title()
            
            # P2CP
            if art in results['p2cp']:
                r = results['p2cp'][art]
                p2cp_mean_std = f"{r['mean']:.2f}±{r['std']:.2f}"
                p2cp_range = f"[{r['min']:.2f}, {r['max']:.2f}]"
            else:
                p2cp_mean_std = p2cp_range = "-"
            
            # Jaccard
            if art in results['jaccard']:
                r = results['jaccard'][art]
                jacc_mean_std = f"{r['mean']:.2f}±{r['std']:.2f}"
                jacc_range = f"[{r['min']:.2f}, {r['max']:.2f}]"
            else:
                jacc_mean_std = jacc_range = "-"
            
            print(f"{art_name:<25} {p2cp_mean_std:<20} {p2cp_range:<15} {jacc_mean_std:<20} {jacc_range:<15}")
        
        # Overall
        print("-" * 100)
        r_p2cp = results['overall_p2cp']
        r_jacc = results['overall_jaccard']
        
        if r_p2cp:
            p2cp_mean_std = f"{r_p2cp['mean']:.2f}±{r_p2cp['std']:.2f}"
            p2cp_range = f"[{r_p2cp['min']:.2f}, {r_p2cp['max']:.2f}]"
        else:
            p2cp_mean_std = p2cp_range = "-"
        
        if r_jacc:
            jacc_mean_std = f"{r_jacc['mean']:.2f}±{r_jacc['std']:.2f}"
            jacc_range = f"[{r_jacc['min']:.2f}, {r_jacc['max']:.2f}]"
        else:
            jacc_mean_std = jacc_range = "-"
        
        print(f"{'OVERALL':<25} {p2cp_mean_std:<20} {p2cp_range:<15} {jacc_mean_std:<20} {jacc_range:<15}")
        
        # Summary
        print(f"\n{'='*100}")
        print(f"Total measurements: {results['count']}")
        if 'sequences' in results:
            print(f"Sequences: {results['sequences']}")
        print(f"{'='*100}\n")
    
    @staticmethod
    def print_single_latex_table(results: Dict):
        """Print LaTeX format table for single subject/sequence"""
        label = results['label']
        
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{lccc}")
        print("\\hline")
        print(f"& \\multicolumn{{2}}{{c}}{{P2CP$_{{RMS}}$ (mm)}} & Jaccard index \\\\")
        print(f"Articulator & Mean $\\pm$ Std & Range & Mean $\\pm$ Std \\\\")
        print("\\hline")
        
        # Per articulator
        all_articulators = set(results['p2cp'].keys()) | set(results['jaccard'].keys())
        for art in sorted(all_articulators):
            art_name = art.replace('-', ' ').title()
            if 'Midline' in art_name:
                art_name = art_name.replace('Soft Palate Midline', 'Soft Palate\\newline Center line')
            
            # P2CP
            if art in results['p2cp']:
                r = results['p2cp'][art]
                p2cp_mean = f"{r['mean']:.2f} $\\pm$ {r['std']:.2f}"
                p2cp_range = f"[{r['min']:.2f}, {r['max']:.2f}]"
            else:
                p2cp_mean = p2cp_range = ""
            
            # Jaccard
            if art in results['jaccard']:
                r = results['jaccard'][art]
                jacc_mean = f"{r['mean']:.2f} $\\pm$ {r['std']:.2f}"
            else:
                jacc_mean = ""
            
            print(f"{art_name} & {p2cp_mean} & {p2cp_range} & {jacc_mean} \\\\")
        
        # Overall
        print("\\hline")
        r_p2cp = results['overall_p2cp']
        r_jacc = results['overall_jaccard']
        
        if r_p2cp:
            p2cp_mean = f"{r_p2cp['mean']:.2f} $\\pm$ {r_p2cp['std']:.2f}"
            p2cp_range = f"[{r_p2cp['min']:.2f}, {r_p2cp['max']:.2f}]"
        else:
            p2cp_mean = p2cp_range = ""
        
        if r_jacc:
            jacc_mean = f"{r_jacc['mean']:.2f} $\\pm$ {r_jacc['std']:.2f}"
        else:
            jacc_mean = ""
        
        print(f"mean $\\pm$ std & {p2cp_mean} & {p2cp_range} & {jacc_mean} \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print(f"\\caption{{Statistics for {label}}}")
        print("\\end{table}")
    
    @staticmethod
    def print_single_markdown_table(results: Dict):
        """Print Markdown format table for single subject/sequence"""
        label = results['label']
        
        print(f"\n## Statistics: {label}\n")
        print("| Articulator | P2CP_RMS (mm) | | Jaccard Index |")
        print("|-------------|---------------|-----------------|---------------|")
        print("|             | Mean ± Std | Range | Mean ± Std |")
        print("|-------------|---------------|-----------------|---------------|")
        
        # Per articulator
        all_articulators = set(results['p2cp'].keys()) | set(results['jaccard'].keys())
        for art in sorted(all_articulators):
            art_name = art.replace('-', ' ').title()
            
            # P2CP
            if art in results['p2cp']:
                r = results['p2cp'][art]
                p2cp_mean = f"{r['mean']:.2f} ± {r['std']:.2f}"
                p2cp_range = f"[{r['min']:.2f}, {r['max']:.2f}]"
            else:
                p2cp_mean = p2cp_range = "-"
            
            # Jaccard
            if art in results['jaccard']:
                r = results['jaccard'][art]
                jacc_mean = f"{r['mean']:.2f} ± {r['std']:.2f}"
            else:
                jacc_mean = "-"
            
            print(f"| {art_name} | {p2cp_mean} | {p2cp_range} | {jacc_mean} |")
        
        # Overall
        r_p2cp = results['overall_p2cp']
        r_jacc = results['overall_jaccard']
        
        if r_p2cp:
            p2cp_mean = f"{r_p2cp['mean']:.2f} ± {r_p2cp['std']:.2f}"
            p2cp_range = f"[{r_p2cp['min']:.2f}, {r_p2cp['max']:.2f}]"
        else:
            p2cp_mean = p2cp_range = "-"
        
        if r_jacc:
            jacc_mean = f"{r_jacc['mean']:.2f} ± {r_jacc['std']:.2f}"
        else:
            jacc_mean = "-"
        
        print(f"| mean ± std | {p2cp_mean} | {p2cp_range} | {jacc_mean} |")
        
        print(f"\nTotal measurements: {results['count']}")
        if 'sequences' in results:
            print(f"Sequences: {', '.join(results['sequences'])}")


def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Object-Oriented Results Table Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and view data summary
  python %(prog)s /path/to/test_results.csv
  
  # Compare two subjects
  python %(prog)s /path/to/test_results.csv --compare subject -s1 1618 -s2 1640
  
  # Compare two sequences
  python %(prog)s /path/to/test_results.csv --compare sequence -s1 S10 -s2 S14
  
  # Generate LaTeX table
  python %(prog)s /path/to/test_results.csv --compare subject -s1 1618 -s2 1640 -f latex
        """
    )
    
    parser.add_argument('csv_path', help='Path to test_results.csv file')
    parser.add_argument('--compare', '--compare-by', '-c',
                        choices=['subject', 'sequence'],
                        help='Comparison mode: subject or sequence level')
    parser.add_argument('--subject1', '-s1', '--item1',
                        help='First subject/sequence to compare')
    parser.add_argument('--subject2', '-s2', '--item2',
                        help='Second subject/sequence to compare')
    parser.add_argument('--format', '-f',
                        choices=['text', 'latex', 'markdown'],
                        default='text',
                        help='Output format (default: text)')
    parser.add_argument('--output', '-o',
                        help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    # Redirect output if specified
    if args.output:
        sys.stdout = open(args.output, 'w')
    
    try:
        # Load data
        data_loader = ResultsDataLoader(args.csv_path)
        engine = ComparisonEngine(data_loader)
        formatter = TableFormatter()
        
        # CASE 1: Single subject dataset - auto-generate stats
        if len(data_loader.subjects) == 1 and not args.compare:
            print("\n✅ Single subject detected - generating statistics report...\n")
            subject_id = data_loader.subjects[0]
            results = engine.generate_single_stats(subject_id, compare_by='subject')
            
            if results:
                if args.format == 'latex':
                    formatter.print_single_latex_table(results)
                elif args.format == 'markdown':
                    formatter.print_single_markdown_table(results)
                else:
                    formatter.print_single_text_table(results)
            return
        
        # CASE 2: Multiple subjects/sequences - need comparison mode
        if not args.compare:
            print("\n✅ Data loaded successfully!")
            print("\n💡 To generate comparison reports, use:")
            if len(data_loader.subjects) > 1:
                subj_example = f"{data_loader.subjects[0]} -s2 {data_loader.subjects[1]}"
                print(f"   --compare subject -s1 {subj_example}")
            if len(data_loader.sequences) > 1:
                seq_example = f"{data_loader.sequences[0]} -s2 {data_loader.sequences[1]}"
                print(f"   --compare sequence -s1 {seq_example}")
            return
        
        # CASE 3: Comparison mode requested
        # Validate comparison arguments
        if not args.subject1 or not args.subject2:
            print("Error: Both --subject1 and --subject2 required for comparison")
            sys.exit(1)
        
        # Convert subject IDs to int if comparing subjects
        if args.compare == 'subject':
            try:
                item1 = int(args.subject1)
                item2 = int(args.subject2)
            except ValueError:
                print(f"Error: Subject IDs must be integers")
                sys.exit(1)
        else:
            item1 = args.subject1
            item2 = args.subject2
        
        # Perform comparison
        results = engine.compare(item1, item2, args.compare)
        
        if results is None:
            sys.exit(1)
        
        # Format and print results
        if args.format == 'latex':
            formatter.print_latex_table(results)
        elif args.format == 'markdown':
            formatter.print_markdown_table(results)
        else:
            formatter.print_text_table(results)
        
    finally:
        if args.output:
            sys.stdout.close()
            print(f"\n✅ Report saved to: {args.output}", file=sys.__stdout__)


if __name__ == '__main__':
    main()
