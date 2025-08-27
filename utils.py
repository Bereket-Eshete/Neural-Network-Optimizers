import numpy as np
import matplotlib.pyplot as plt

def plot_comparison_figure(history_gd, history_adam):
    """
    Create a single figure with 4 subplots comparing GD and Adam optimizers
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training Loss
    epochs = range(len(history_gd['train_loss']))
    ax1.plot(epochs, history_gd['train_loss'], 'b-', label='Gradient Descent', linewidth=2)
    ax1.plot(epochs, history_adam['train_loss'], 'r-', label='Adam', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2.plot(epochs, history_gd['val_loss'], 'b-', label='Gradient Descent', linewidth=2)
    ax2.plot(epochs, history_adam['val_loss'], 'r-', label='Adam', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax3.plot(epochs, history_gd['train_accuracy'], 'b-', label='Gradient Descent', linewidth=2)
    ax3.plot(epochs, history_adam['train_accuracy'], 'r-', label='Adam', linewidth=2)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Training Accuracy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy
    ax4.plot(epochs, history_gd['val_accuracy'], 'b-', label='Gradient Descent', linewidth=2)
    ax4.plot(epochs, history_adam['val_accuracy'], 'r-', label='Adam', linewidth=2)
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Validation Accuracy Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_newton_comparison(history_adam, history_newton):
    """
    Create comparison plots between Adam and Newton's Method
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training Loss
    epochs_adam = range(len(history_adam['train_loss']))
    epochs_newton = range(len(history_newton['train_loss']))
    
    ax1.plot(epochs_adam, history_adam['train_loss'], 'b-', label='Adam', linewidth=2)
    ax1.plot(epochs_newton, history_newton['train_loss'], 'r-', label="Newton's Method", linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss: Adam vs Newton')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2.plot(epochs_adam, history_adam['val_loss'], 'b-', label='Adam', linewidth=2)
    ax2.plot(epochs_newton, history_newton['val_loss'], 'r-', label="Newton's Method", linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss: Adam vs Newton')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax3.plot(epochs_adam, history_adam['train_accuracy'], 'b-', label='Adam', linewidth=2)
    ax3.plot(epochs_newton, history_newton['train_accuracy'], 'r-', label="Newton's Method", linewidth=2)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Training Accuracy: Adam vs Newton')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy
    ax4.plot(epochs_adam, history_adam['val_accuracy'], 'b-', label='Adam', linewidth=2)
    ax4.plot(epochs_newton, history_newton['val_accuracy'], 'r-', label="Newton's Method", linewidth=2)
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Validation Accuracy: Adam vs Newton')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_dynamic_report(history_gd, history_adam, test_accuracy_gd, test_accuracy_adam):
    """
    Generate a dynamic report based on actual results for GD vs Adam
    """
    print("=" * 60)
    print("GRADIENT DESCENT vs ADAM COMPARISON REPORT")
    print("=" * 60)
    
    # Calculate improvements
    gd_train_improvement = (history_gd['train_accuracy'][-1] - history_gd['train_accuracy'][0]) * 100
    adam_train_improvement = (history_adam['train_accuracy'][-1] - history_adam['train_accuracy'][0]) * 100
    
    gd_val_improvement = (history_gd['val_accuracy'][-1] - history_gd['val_accuracy'][0]) * 100
    adam_val_improvement = (history_adam['val_accuracy'][-1] - history_adam['val_accuracy'][0]) * 100
    
    # Determine which optimizer performed better
    gd_better_train = history_gd['train_accuracy'][-1] > history_adam['train_accuracy'][-1]
    gd_better_val = history_gd['val_accuracy'][-1] > history_adam['val_accuracy'][-1]
    gd_better_test = test_accuracy_gd > test_accuracy_adam
    
    print(f"\nFinal Training Loss:")
    print(f"  Gradient Descent: {history_gd['train_loss'][-1]:.4f}")
    print(f"  Adam: {history_adam['train_loss'][-1]:.4f}")
    
    print(f"\nFinal Validation Loss:")
    print(f"  Gradient Descent: {history_gd['val_loss'][-1]:.4f}")
    print(f"  Adam: {history_adam['val_loss'][-1]:.4f}")
    
    print(f"\nFinal Training Accuracy:")
    print(f"  Gradient Descent: {history_gd['train_accuracy'][-1]:.4f} (+{gd_train_improvement:.1f}%)")
    print(f"  Adam: {history_adam['train_accuracy'][-1]:.4f} (+{adam_train_improvement:.1f}%)")
    
    print(f"\nFinal Validation Accuracy:")
    print(f"  Gradient Descent: {history_gd['val_accuracy'][-1]:.4f} (+{gd_val_improvement:.1f}%)")
    print(f"  Adam: {history_adam['val_accuracy'][-1]:.4f} (+{adam_val_improvement:.1f}%)")
    
    print(f"\nTest Accuracy:")
    print(f"  Gradient Descent: {test_accuracy_gd:.4f}")
    print(f"  Adam: {test_accuracy_adam:.4f}")
    
    print(f"\nPerformance Summary:")
    print(f"  Training: {'Gradient Descent' if gd_better_train else 'Adam'} performed better")
    print(f"  Validation: {'Gradient Descent' if gd_better_val else 'Adam'} performed better")
    print(f"  Test: {'Gradient Descent' if gd_better_test else 'Adam'} performed better")
    
    print(f"\nKey Observations:")
    if gd_better_val and gd_better_test:
        print("- Gradient Descent showed better generalization performance")
        print("- Adam may need hyperparameter tuning for this dataset")
    else:
        print("- Adam demonstrated faster convergence as expected")
        print("- Both optimizers achieved competitive results")
    
    # Convergence speed analysis
    try:
        gd_convergence_epoch = next(i for i, acc in enumerate(history_gd['val_accuracy']) if acc > 0.7)
        adam_convergence_epoch = next(i for i, acc in enumerate(history_adam['val_accuracy']) if acc > 0.7)
        print(f"- Gradient Descent reached 70% validation accuracy at epoch {gd_convergence_epoch}")
        print(f"- Adam reached 70% validation accuracy at epoch {adam_convergence_epoch}")
    except:
        print("  Could not determine 70% convergence point")
    
    print(f"\nRecommendations:")
    if gd_better_val:
        print("- Consider using Gradient Descent with learning rate scheduling")
        print("- For Adam, try lower learning rate or adjust beta parameters")
    else:
        print("- Adam's adaptive learning rate proved effective for this problem")
        print("- Both optimizers are viable choices for this task")

def generate_newton_report(history_adam, history_newton, test_accuracy_adam, test_accuracy_newton):
    """
    Generate a comparison report between Adam and Newton's Method
    """
    print("=" * 70)
    print("NEWTON'S METHOD vs ADAM COMPARISON REPORT")
    print("=" * 70)
    
    # Calculate final metrics
    final_train_loss_adam = history_adam['train_loss'][-1]
    final_train_loss_newton = history_newton['train_loss'][-1]
    final_val_loss_adam = history_adam['val_loss'][-1]
    final_val_loss_newton = history_newton['val_loss'][-1]
    final_train_acc_adam = history_adam['train_accuracy'][-1]
    final_train_acc_newton = history_newton['train_accuracy'][-1]
    final_val_acc_adam = history_adam['val_accuracy'][-1]
    final_val_acc_newton = history_newton['val_accuracy'][-1]
    
    print(f"\nFinal Training Loss:")
    print(f"  Adam: {final_train_loss_adam:.4f}")
    print(f"  Newton's Method: {final_train_loss_newton:.4f}")
    
    print(f"\nFinal Validation Loss:")
    print(f"  Adam: {final_val_loss_adam:.4f}")
    print(f"  Newton's Method: {final_val_loss_newton:.4f}")
    
    print(f"\nFinal Training Accuracy:")
    print(f"  Adam: {final_train_acc_adam:.4f}")
    print(f"  Newton's Method: {final_train_acc_newton:.4f}")
    
    print(f"\nFinal Validation Accuracy:")
    print(f"  Adam: {final_val_acc_adam:.4f}")
    print(f"  Newton's Method: {final_val_acc_newton:.4f}")
    
    print(f"\nTest Accuracy:")
    print(f"  Adam: {test_accuracy_adam:.4f}")
    print(f"  Newton's Method: {test_accuracy_newton:.4f}")
    
    print(f"\nKey Observations:")
    
    # Performance comparison
    if test_accuracy_newton > test_accuracy_adam:
        print("- Newton's Method achieved better final performance")
    else:
        print("- Adam achieved better final performance")
    
    # Convergence analysis
    if len(history_newton['train_loss']) < len(history_adam['train_loss']):
        print("- Newton's Method showed faster convergence per epoch")
    else:
        print("- Adam showed more stable convergence across epochs")
    
    print("- Newton's Method uses second-order information (curvature)")
    print("- Adam uses first-order gradients with adaptive learning rates")
    print("- Newton's Method is computationally expensive but theoretically optimal")
    
    print(f"\nComputational Considerations:")
    print("- Newton's Method requires O(n²) operations for Hessian computation")
    print("- Adam requires O(n) operations per update")
    print("- Newton's Method is better suited for smaller networks or problems")
    
    print(f"\nFinal Verdict:")
    if abs(test_accuracy_adam - test_accuracy_newton) < 0.005:
        print("✓ Both methods performed comparably well")
    elif test_accuracy_newton > test_accuracy_adam:
        print("✓ Newton's Method demonstrated superior performance")
    else:
        print("✓ Adam demonstrated superior performance")

def save_plots(fig_gd_adam, fig_adam_newton):
    """Save both comparison plots"""
    fig_gd_adam.savefig('gd_vs_adam_comparison.png', dpi=300, bbox_inches='tight')
    fig_adam_newton.savefig('adam_vs_newton_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plots saved as 'gd_vs_adam_comparison.png' and 'adam_vs_newton_comparison.png'")