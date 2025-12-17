"""
COMPREHENSIVE Model Evaluation Module for ASL Recognition - PHASE 1 & 2 ONLY
Evaluates all models on test set using Phase 1 and Phase 2 weights only
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import warnings
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
    top_k_accuracy_score
)

# Filter warnings
warnings.filterwarnings('ignore')

# Import configurations
try:
    from config import (
        DATA_DIR, SAVED_MODELS_DIR, LOGS_DIR, DOCS_DIR,
        IMG_SIZE, BATCH_SIZE, NUM_CLASSES
    )
    from model_builder import ModelBuilder
    from data_preprocessing import get_preprocessing_settings
except ImportError:
    print("⚠️ Warning: Could not import config, using defaults")
    DATA_DIR = Path(r'D:\asl_project\data\real_world_test')
    SAVED_MODELS_DIR = Path('./models/saved_models')
    LOGS_DIR = Path('./models/training_logs')
    DOCS_DIR = Path('./docs')
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 27


class PhaseEvaluator:
    """Evaluates trained models on test dataset using Phase 1 and Phase 2 weights"""

    def __init__(self, 
                 model_names: List[str] = ['ResNet50', 'EfficientNetB0', 'InceptionV3'], 
                 phases: List[int] = [1, 2],
                 data_dir: Optional[Path] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_names: List of model names to evaluate.
            phases: List of phases to evaluate.
            data_dir: Optional override for data directory.
        """
        self.model_names = model_names
        self.phases = phases
        self.data_dir = data_dir if data_dir else DATA_DIR
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.test_generator = None
        self.top_k = 5

        # Create directories
        self.eval_dir = DOCS_DIR / 'evaluation_phases'
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Load class indices
        self.class_indices = self._load_class_indices()
        self.class_names = list(self.class_indices.values()) if self.class_indices else []

        print("\n" + "🎯" * 50)
        print("ASL Recognition - PHASE 1 & 2 MODEL EVALUATION SYSTEM")
        print("🎯" * 50)
        print(f"\n📌 This system evaluates models using Phase {self.phases} weights")
        print("   Expected weights files: 'modelname_phase1_best.weights.h5' and 'modelname_phase2_best.weights.h5'\n")

    def _load_class_indices(self) -> Dict[int, str]:
        """Load class indices from file"""
        class_indices_path = self.data_dir / 'class_indices.json'
        # Fallback to config DATA_DIR if specific data_dir doesn't have it (optional safety)
        if not class_indices_path.exists():
             class_indices_path = DATA_DIR / 'class_indices.json'

        if class_indices_path.exists():
            try:
                with open(class_indices_path, 'r') as f:
                    indices = json.load(f)
                # Convert to {index: class_name}
                return {v: k for k, v in indices.items()}
            except Exception as e:
                print(f"⚠️ Error loading class indices: {e}")
        
        print("⚠️ Using default class names (A-Z + del, nothing, space)")
        class_names = [chr(65 + i) for i in range(26)] + ['del', 'nothing', 'space']
        return {i: class_names[i] for i in range(len(class_names))}

    def check_phase_weights(self) -> Tuple[Dict[str, Dict[int, bool]], List[str]]:
        """Check if Phase 1 and Phase 2 weights exist for all models"""
        print(f"🔍 Checking for Phase {self.phases} weights...")

        available_models = {}
        missing_models = []

        for model_name in self.model_names:
            available_models[model_name] = {}
            for phase in self.phases:
                phase_path = SAVED_MODELS_DIR / f'{model_name}_phase{phase}_best.weights.h5'
                exists = phase_path.exists()
                available_models[model_name][phase] = exists
                status_icon = "✅" if exists else "❌"
                status_text = "Found" if exists else "NOT FOUND"
                print(f"   {status_icon} {model_name} Phase {phase}: {status_text}")
            
            # Check if model has at least one phase available
            if not any(available_models[model_name].values()):
                missing_models.append(model_name)

        print(f"\n📊 Summary:")
        models_with_weights = [m for m in self.model_names if any(available_models[m].values())]
        print(f"   Models with weights: {len(models_with_weights)}")
        print(f"   Missing models: {len(missing_models)}")

        return available_models, missing_models

    def load_test_generator(self, model_name: str = 'ResNet50') -> Any:
        """
        Load test data generator for a specific model.
        
        Args:
            model_name: Name of the model to determine preprocessing.
            
        Returns:
            Keras ImageDataGenerator flow.
        """
        print(f"\n📊 Loading test data generator for {model_name}...")

        # Get preprocessing function
        preprocessing_function, rescale_value = get_preprocessing_settings(model_name)

        # Create test generator
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            rescale=rescale_value
        )

        test_dir = self.data_dir / 'test'
        if not test_dir.exists():
            print(f"❌ Test directory not found: {test_dir}")
            # Try fallback to DATA_DIR/test if self.data_dir was custom
            if (DATA_DIR / 'test').exists():
                 test_dir = DATA_DIR / 'test'
                 print(f"⚠️ Falling back to: {test_dir}")
            else:
                 raise FileNotFoundError(f"Test directory not found at {test_dir}")

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False  # Important for evaluation!
        )

        print(f"✅ Test samples: {test_generator.samples}")
        print(f"✅ Classes: {test_generator.num_classes}")

        self.test_generator = test_generator
        return test_generator

    def load_phase_model(self, model_name: str, phase: int) -> Optional[tf.keras.Model]:
        """Load a trained model with Phase 1 or Phase 2 weights"""
        print(f"\n🔍 Loading {model_name} Phase {phase} model...")

        phase_path = SAVED_MODELS_DIR / f'{model_name}_phase{phase}_best.weights.h5'

        if not phase_path.exists():
            print(f"❌ ERROR: Phase {phase} weights not found for {model_name}")
            print(f"   Expected: {phase_path}")
            return None

        try:
            # Build model
            builder = ModelBuilder(model_name)
            model = builder.build_model()

            # For Phase 2, need to unfreeze layers
            if phase == 2:
                builder.unfreeze_for_finetuning(phase=2)

            # Load weights
            print(f"📥 Loading Phase {phase} weights: {phase_path.name}")
            model.load_weights(phase_path)

            # Store information
            model_key = f'{model_name}_phase{phase}'
            self.models[model_key] = {
                'model': model,
                'builder': builder,
                'weights_path': phase_path,
                'phase': phase,
                'model_name': model_name
            }

            # Verify weights are loaded (simple check)
            trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

            print(f"✅ {model_name} Phase {phase} loaded successfully!")
            print(f"   Test Set Parameters:")
            print(f"   - Total parameters: {model.count_params():,}")
            print(f"   - Trainable parameters: {trainable_params:,}")

            return model

        except Exception as e:
            print(f"❌ Error loading {model_name} Phase {phase}: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_single_model(self, model_name: str, phase: int) -> Optional[Dict[str, Any]]:
        """Evaluate a single model on test set for a specific phase"""
        model_key = f'{model_name}_phase{phase}'
        if model_key not in self.models:
            print(f"❌ Model {model_name} Phase {phase} not loaded!")
            return None

        print(f"\n{'=' * 60}")
        print(f"🧪 Evaluating {model_name} PHASE {phase} on Test Set")
        print(f"{'=' * 60}")

        model = self.models[model_key]['model']

        # Load test generator if not already loaded or if needed for a different model type/preprocessing
        # Note: In a robust system, we should check if preprocessing matches the current model.
        # For this implementation, we reload if model type changes usually, but here we just load.
        self.load_test_generator(model_name)

        # Reset generator
        self.test_generator.reset()

        # Predict
        print("📈 Making predictions...")
        # verbose=1 for progress bar
        y_pred_proba = model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # True labels
        y_true = self.test_generator.classes

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Top-K Accuracy
        try:
            top_k_acc = top_k_accuracy_score(y_true, y_pred_proba, k=self.top_k)
        except Exception as e:
            print(f"⚠️ Could not calculate Top-{self.top_k} accuracy: {e}")
            top_k_acc = 0.0

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        # Detailed classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names[:len(np.unique(y_true))],
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Store results
        result_key = f'{model_name}_phase{phase}'
        self.results[result_key] = {
            'model_name': model_name,
            'phase': phase,
            'accuracy': accuracy,
            'top_k_accuracy': top_k_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': report,
            'test_samples': len(y_true),
            'misclassified': np.sum(y_true != y_pred)
        }

        # Print summary
        print(f"\n📊 {model_name} PHASE {phase} Results:")
        print(f"   Accuracy:         {accuracy:.4f}")
        print(f"   Top-{self.top_k} Accuracy:  {top_k_acc:.4f}")
        print(f"   Precision:        {precision:.4f}")
        print(f"   Recall:           {recall:.4f}")
        print(f"   F1-Score:         {f1:.4f}")
        print(f"   Error Rate:       {np.sum(y_true != y_pred) / len(y_true) * 100:.2f}%")

        # Save detailed report
        self._save_detailed_report(model_name, phase, report)

        return self.results[result_key]

    def _save_detailed_report(self, model_name: str, phase: int, report: Dict) -> None:
        """Save detailed classification report"""
        report_path = self.eval_dir / f'{model_name}_phase{phase}_detailed_report.json'

        # Convert numpy arrays to lists for JSON serialization
        report_serializable = {}
        for key, value in report.items():
            if isinstance(value, dict):
                report_serializable[key] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in value.items()
                }
            elif isinstance(value, (np.float32, np.float64)):
                report_serializable[key] = float(value)
            else:
                report_serializable[key] = value

        with open(report_path, 'w') as f:
            json.dump(report_serializable, f, indent=4)

        print(f"✅ Detailed report saved: {report_path}")

    def plot_confusion_matrix(self, model_name: str, phase: int, figsize: Tuple[int, int] = (16, 12)) -> Optional[Path]:
        """Plot confusion matrix for a specific phase model"""
        result_key = f'{model_name}_phase{phase}'
        if result_key not in self.results:
            print(f"❌ No results for {model_name} Phase {phase}")
            return None

        cm = self.results[result_key]['confusion_matrix']

        # 1. Normalized Heatmap
        plt.figure(figsize=figsize)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names[:cm.shape[1]],
            yticklabels=self.class_names[:cm.shape[0]],
            cbar_kws={'label': 'Normalized Accuracy'}
        )

        plt.title(
            f'{model_name} - Phase {phase} Confusion Matrix\n(Test Set, {self.results[result_key]["test_samples"]} samples)',
            fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        cm_path = self.eval_dir / f'{model_name}_phase{phase}_confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Confusion matrix saved: {cm_path}")

        # 2. Counts Heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Greens',
            xticklabels=self.class_names[:cm.shape[1]],
            yticklabels=self.class_names[:cm.shape[0]]
        )
        plt.title(f'{model_name} - Phase {phase} Confusion Matrix (Counts)',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        cm_counts_path = self.eval_dir / f'{model_name}_phase{phase}_confusion_matrix_counts.png'
        plt.savefig(cm_counts_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Confusion matrix (counts) saved: {cm_counts_path}")
        return cm_path

    def plot_model_comparison(self, phase: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Create comparison plots for models (all phases or specific phase)"""
        if len(self.results) < 1:
            print("❌ Need to evaluate at least 1 model first")
            return None

        # Prepare comparison data
        comparison_data = []
        for result_key, result in self.results.items():
            # Filter by phase if specified
            if phase is not None and result['phase'] != phase:
                continue
            
            comparison_data.append({
                'Model': f"{result['model_name']} Phase {result['phase']}",
                'Model_Name': result['model_name'],
                'Phase': result['phase'],
                'Accuracy': result['accuracy'],
                'Top5_Accuracy': result.get('top_k_accuracy', 0.0),
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'Test_Samples': result['test_samples'],
                'Misclassified': result['misclassified'],
                'Error_Rate': result['misclassified'] / result['test_samples'] * 100
            })

        df_comparison = pd.DataFrame(comparison_data)
        if df_comparison.empty:
            print(f"⚠️ No data to compare for Phase {phase}")
            return None

        # Plot 1: Bar chart comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        metrics = ['Accuracy', 'Top5_Accuracy', 'Recall', 'F1-Score']
        colors = ['#3498DB', '#9B59B6', '#E74C3C', '#F39C12']

        phase_title = f"Phase {phase}" if phase else "All Phases"
        for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
            bars = ax.bar(df_comparison['Model'], df_comparison[metric],
                          color=color, alpha=0.8)
            ax.set_title(f'{metric} Comparison ({phase_title})', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=10)

            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

        plt.suptitle(f'{phase_title} Models Performance Comparison',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        phase_suffix = f"_phase{phase}" if phase else "_all_phases"
        comparison_path = self.eval_dir / f'phases{phase_suffix}_model_comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ {phase_title} model comparison saved: {comparison_path}")

        # Additional plots
        self._plot_radar_chart(df_comparison, phase)
        self._plot_error_comparison(df_comparison, phase)

        return df_comparison

    def _plot_radar_chart(self, df_comparison: pd.DataFrame, phase: Optional[int] = None) -> None:
        """Create radar chart for model comparison"""
        metrics = ['Accuracy', 'Top5_Accuracy', 'Precision', 'Recall', 'F1-Score']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Use a colormap
        colors = plt.cm.tab10(np.linspace(0, 1, len(df_comparison)))

        for idx, row in df_comparison.iterrows():
            values = row[metrics].values.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim([0, 1])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        phase_title = f"Phase {phase}" if phase else "All Phases"
        plt.title(f'{phase_title} Models Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)

        phase_suffix = f"_phase{phase}" if phase else "_all_phases"
        radar_path = self.eval_dir / f'phases{phase_suffix}_radar_comparison.png'
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ {phase_title} radar chart saved: {radar_path}")

    def _plot_error_comparison(self, df_comparison: pd.DataFrame, phase: Optional[int] = None) -> None:
        """Plot error rate comparison for models"""
        phase_title = f"Phase {phase}" if phase else "All Phases"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Misclassified counts
        bars1 = ax1.bar(df_comparison['Model'], df_comparison['Misclassified'],
                        color='#E74C3C', alpha=0.8)
        ax1.set_title(f'Misclassified Samples ({phase_title})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        # Plot 2: Error rate percentage
        bars2 = ax2.bar(df_comparison['Model'], df_comparison['Error_Rate'],
                        color='#F39C12', alpha=0.8)
        ax2.set_title(f'Error Rate % ({phase_title})', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Error Rate (%)')
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=15, ha='right')

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

        plt.suptitle(f'{phase_title} Models Error Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        phase_suffix = f"_phase{phase}" if phase else "_all_phases"
        error_path = self.eval_dir / f'phases{phase_suffix}_error_analysis.png'
        plt.savefig(error_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ {phase_title} error analysis saved: {error_path}")

    def generate_summary_report(self) -> Optional[pd.DataFrame]:
        """Generate a comprehensive summary report for all evaluated phases"""
        if not self.results:
            print("❌ No evaluation results available")
            return None

        print(f"\n{'=' * 60}")
        print("📋 GENERATING COMPREHENSIVE EVALUATION REPORT")
        print(f"{'=' * 60}")

        # Create summary dataframe
        summary_data = []
        for result_key, result in self.results.items():
            report = result['classification_report']
            weighted_avg = report.get('weighted avg', {})

            summary_data.append({
                'Model': result['model_name'],
                'Phase': result['phase'],
                'Accuracy': result['accuracy'],
                'Top5_Acc': result.get('top_k_accuracy', 0.0),
                'Precision': weighted_avg.get('precision', 0),
                'Recall': weighted_avg.get('recall', 0),
                'F1-Score': weighted_avg.get('f1-score', 0),
                'Test_Samples': result['test_samples'],
                'Misclassified': result['misclassified'],
                'Error_Rate_%': (result['misclassified'] / result['test_samples']) * 100
            })

        df_summary = pd.DataFrame(summary_data)

        # Add ranking - simple combined score
        df_summary['Combined_Score'] = df_summary['Accuracy'] + df_summary['F1-Score'] + (df_summary['Top5_Acc'] * 0.5)
        df_summary['Overall_Rank'] = df_summary['Combined_Score'].rank(ascending=False).astype(int)
        
        # Sort by overall rank
        df_summary = df_summary.sort_values('Overall_Rank')

        # Save summary
        summary_path = self.eval_dir / 'phases_evaluation_summary.csv'
        df_summary.to_csv(summary_path, index=False)
        
        df_summary.to_json(self.eval_dir / 'phases_evaluation_summary.json', orient='records', indent=4)

        # Print summary table
        print("\n📊 MODEL EVALUATION SUMMARY:")
        print("=" * 90)
        print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
        print("=" * 90)

        # Find best model
        best_model = df_summary.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model['Model']} Phase {best_model['Phase']}")
        print(f"   Accuracy:     {best_model['Accuracy']:.4f}")
        print(f"   Top-5 Acc:    {best_model['Top5_Acc']:.4f}")
        print(f"   F1-Score:     {best_model['F1-Score']:.4f}")

        # Save best model info
        best_model_path = self.eval_dir / 'phases_best_model_info.json'
        with open(best_model_path, 'w') as f:
            json.dump(best_model.to_dict(), f, indent=4)

        return df_summary

    def save_all_results(self) -> None:
        """Save all results and plots"""
        print(f"\n💾 Saving all evaluation results...")

        results_path = self.eval_dir / 'phases_all_results.json'

        # Convert to serializable
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in result.items()
                if k not in ['y_pred_proba', 'confusion_matrix']
            }

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)

        print(f"✅ All results saved: {results_path}")

    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all models with Phase 1 and Phase 2 weights"""
        print(f"\n{'🚀' * 30}")
        print("EVALUATING ALL MODELS WITH PHASE 1 & 2 WEIGHTS")
        print(f"{'🚀' * 30}")

        # Check for weights
        available_models, missing_models = self.check_phase_weights()

        if not available_models:
            print("\n❌ CRITICAL ERROR: No Phase 1 or Phase 2 weights found!")
            return {}

        all_results = {}

        # Evaluate loop
        for model_name in self.model_names:
            for phase in self.phases:
                model_key = f'{model_name}_phase{phase}'
                
                if model_name in available_models and available_models[model_name].get(phase, False):
                    print(f"\n{'=' * 60}")
                    print(f"PROCESSING: {model_name} (Phase {phase})")
                    print(f"{'=' * 60}")

                    model = self.load_phase_model(model_name, phase)
                    if model is None:
                        continue

                    result = self.evaluate_single_model(model_name, phase)
                    if result:
                        all_results[model_key] = result
                    
                    self.plot_confusion_matrix(model_name, phase)
                    
                    # Clear session to free memory
                    tf.keras.backend.clear_session()
                else:
                    if model_name in missing_models:
                        pass # Already warned in summary
                    else:
                        print(f"⚠️ {model_name} Phase {phase} skipped (weights missing)")

        # Generate reports if we have results
        if len(all_results) >= 1:
            for phase in self.phases:
                self.plot_model_comparison(phase=phase)
            
            self.plot_model_comparison(phase=None)
            self.generate_summary_report()
            self.save_all_results()

        print(f"\n{'✅' * 30}")
        print("PHASE 1 & 2 EVALUATION COMPLETE!")
        return all_results


def parse_arguments():
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(description="Evaluate ASL Recognition Models")
    
    parser.add_argument('--models', nargs='+', default=['ResNet50', 'EfficientNetB0', 'InceptionV3'],
                        help="List of model names to evaluate")
    parser.add_argument('--phases', nargs='+', type=int, default=[1, 2],
                        help="List of phases to evaluate (e.g. 1 2)")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Override data directory path")
    parser.add_argument('--no-gpu', action='store_true', help="Disable GPU usage")

    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_arguments()

    if args.no_gpu:
        try:
            tf.config.set_visible_devices([], 'GPU')
            print("� GPU disabled by user")
        except:
            pass

    print("\n" + "🎯" * 50)
    print("ASL Recognition - PHASE 1 & 2 MODEL EVALUATION SYSTEM")
    print("🎯" * 50)

    # Resolve Data Dir
    data_dir = Path(args.data_dir) if args.data_dir else None

    # Create evaluator
    evaluator = PhaseEvaluator(model_names=args.models, phases=args.phases, data_dir=data_dir)

    # Evaluate
    results = evaluator.evaluate_all_models()

    # Final Summary
    if results:
        print(f"\n📁 Results saved in: {evaluator.eval_dir}")

if __name__ == "__main__":
    main()