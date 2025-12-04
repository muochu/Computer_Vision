"""Visualization utilities."""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_mhi(mhi, title='MHI', save_path=None):
    """Visualize Motion History Image."""
    plt.figure(figsize=(8, 6))
    plt.imshow(mhi, cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def annotate_frame(frame, label, confidence=None):
    """Annotate frame with predicted label."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    text = label
    if confidence is not None:
        text += f' ({confidence:.2f})'
    
    cv2.putText(frame_bgr, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame_bgr


def create_output_video(frames, labels, output_path, fps=25):
    """Create output video with predicted labels."""
    if not frames:
        return
    
    height, width = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if len(labels) == 1:
        labels = labels * len(frames)
    
    for frame, label in zip(frames, labels):
        annotated = annotate_frame(frame, label)
        out.write(annotated)
    
    out.release()


def visualize_frame_differencing_comparison(frames, theta=30, save_path=None):
    """Visualize frame differencing before and after morphological opening."""
    if len(frames) < 2:
        return
    
    # Import here to avoid circular imports
    from core.mhi import frame_difference, morphological_open
    
    # Get binary mask before morphology
    binary_before = frame_difference(frames[0], frames[1], theta)
    
    # Get binary mask after morphology
    binary_after = morphological_open(binary_before)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(binary_before, cmap='gray')
    axes[0].set_title('Before Morphological Opening', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(binary_after, cmap='gray')
    axes[1].set_title('After Morphological Opening', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_pipeline_diagram(save_path=None):
    """Create a pipeline diagram showing the processing flow."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    
    # Define boxes and arrows
    boxes = [
        ('Input Video', 0.1, 0.5),
        ('Frame\nDifferencing', 0.3, 0.5),
        ('Morphological\nOpening', 0.5, 0.5),
        ('MHI\nConstruction', 0.7, 0.5),
        ('Hu Moments\nExtraction', 0.9, 0.5),
    ]
    
    # Draw boxes
    for text, x, y in boxes:
        box = plt.Rectangle((x-0.08, y-0.15), 0.16, 0.3, 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw arrows
    for i in range(len(boxes) - 1):
        x1, y1 = boxes[i][1] + 0.08, boxes[i][2]
        x2, y2 = boxes[i+1][1] - 0.08, boxes[i+1][2]
        ax.arrow(x1, y1, x2-x1, 0, head_width=0.03, head_length=0.02, 
                fc='black', ec='black', linewidth=1.5)
    
    # Add classifier box below
    classifier_box = plt.Rectangle((0.7, 0.1), 0.2, 0.15, 
                                   facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(0.8, 0.175, 'k-NN\nClassifier', ha='center', va='center', 
           fontsize=10, weight='bold')
    
    # Arrow from Hu Moments to Classifier
    ax.arrow(0.9, 0.35, 0, -0.1, head_width=0.02, head_length=0.015, 
            fc='black', ec='black', linewidth=1.5)
    
    # Output box
    output_box = plt.Rectangle((0.7, 0.0), 0.2, 0.05, 
                               facecolor='yellow', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(0.8, 0.025, 'Action Label', ha='center', va='center', 
           fontsize=9, weight='bold')
    
    # Arrow from Classifier to Output
    ax.arrow(0.8, 0.1, 0, -0.05, head_width=0.02, head_length=0.01, 
            fc='black', ec='black', linewidth=1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_success_case(frames, mhi, true_label, predicted_label, save_path=None):
    """Visualize a successful classification case with frame, MHI, and labels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show sample frame
    axes[0].imshow(frames[len(frames)//2], cmap='gray')
    axes[0].set_title('Sample Frame', fontsize=12)
    axes[0].axis('off')
    
    # Show MHI
    axes[1].imshow(mhi, cmap='gray')
    axes[1].set_title('Motion History Image', fontsize=12)
    axes[1].axis('off')
    
    # Show classification result
    axes[2].axis('off')
    result_text = f'True Label: {true_label}\nPredicted: {predicted_label}\n\n'
    result_text += '✓ Correct Classification'
    axes[2].text(0.5, 0.5, result_text, ha='center', va='center', 
                fontsize=14, weight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    axes[2].set_title('Classification Result', fontsize=12)
    
    plt.suptitle(f'Success Case: {true_label}', fontsize=14, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_failure_case(frames1, mhi1, label1, frames2, mhi2, label2, 
                          predicted1, predicted2, save_path=None):
    """Visualize a failure case comparing two similar actions side-by-side."""
    fig = plt.figure(figsize=(16, 6))
    
    # First action
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(frames1[len(frames1)//2], cmap='gray')
    ax1.set_title(f'Sample Frame: {label1}', fontsize=11)
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(mhi1, cmap='gray')
    ax2.set_title(f'MHI: {label1}', fontsize=11)
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    result_text1 = f'True: {label1}\nPredicted: {predicted1}\n'
    if label1 != predicted1:
        result_text1 += '✗ Misclassified'
        color1 = 'red'
    else:
        result_text1 += '✓ Correct'
        color1 = 'green'
    ax3.text(0.5, 0.5, result_text1, ha='center', va='center', 
            fontsize=12, weight='bold', color=color1,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Second action
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(frames2[len(frames2)//2], cmap='gray')
    ax4.set_title(f'Sample Frame: {label2}', fontsize=11)
    ax4.axis('off')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(mhi2, cmap='gray')
    ax5.set_title(f'MHI: {label2}', fontsize=11)
    ax5.axis('off')
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    result_text2 = f'True: {label2}\nPredicted: {predicted2}\n'
    if label2 != predicted2:
        result_text2 += '✗ Misclassified'
        color2 = 'red'
    else:
        result_text2 += '✓ Correct'
        color2 = 'green'
    ax6.text(0.5, 0.5, result_text2, ha='center', va='center', 
            fontsize=12, weight='bold', color=color2,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.suptitle('Failure Case: Similar Actions (Jogging vs Running)', 
                fontsize=14, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
