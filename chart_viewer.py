"""
Apex Academic Analytics - Standalone Chart Viewer
Displays generated academic charts in an isolated GUI process so that closing
the graph window never terminates the main terminal assistant session.
"""
import sys
import os
import matplotlib
# Use TkAgg on Windows for dependable GUI rendering
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_chart(image_path: str, title: str = "Academic Chart"):
    if not os.path.exists(image_path):
        return
        
    try:
        img = mpimg.imread(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # Calculate optimal window dimensions from image aspect ratio
    h, w = img.shape[:2]
    aspect = w / max(h, 1)
    fig_w = 11.0
    fig_h = max(5.0, min(8.0, fig_w / aspect))
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Set window title
    try:
        fig.canvas.manager.set_window_title(f"Apex Institute of Technology - {title}")
    except Exception:
        pass
        
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Try bringing window to the front
    try:
        manager = fig.canvas.manager
        manager.window.attributes("-topmost", True)
        manager.window.update()
        manager.window.attributes("-topmost", False)
    except Exception:
        pass
        
    # Block ONLY inside this independent subprocess
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        win_title = sys.argv[2] if len(sys.argv) > 2 else "Academic Chart"
        show_chart(img_path, win_title)
