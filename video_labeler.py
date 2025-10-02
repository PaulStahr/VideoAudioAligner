import sys
import os
import numpy as np
import cv2
import pandas as pd
import pyqtgraph as pg
import subprocess
from PyQt6 import QtWidgets, QtCore
import logging
import json

logger = logging.getLogger(__name__)

def export_matrix_as_cube(matrix, size=33, filename="output.cube", clip=False):
    """
    Exports a 3x4 affine color transform matrix as a .cube LUT file.

    Parameters
    ----------
    matrix : np.ndarray
        A 3x4 matrix that transforms [R,G,B,1] into [R',G',B'].
    size : int
        Size of the LUT grid per channel (commonly 17, 33, or 65).
    filename : str
        Output LUT filename (.cube).
    clip : bool
        If True, clamp output values to [0,1]. If False, allow values outside that range.
    """

    if matrix.shape != (3,4):
        raise ValueError("Matrix must be 3x4.")

    # Header
    lines = []
    lines.append("TITLE \"Affine Matrix LUT\"")
    lines.append(f"LUT_3D_SIZE {size}")
    lines.append("DOMAIN_MIN 0.0 0.0 0.0")
    lines.append("DOMAIN_MAX 1.0 1.0 1.0")

    # Create LUT grid and apply transform
    values = np.linspace(0.0, 1.0, size)
    for b in values:
        for g in values:
            for r in values:
                rgb1 = np.array([r, g, b, 1.0])
                out = matrix @ rgb1
                if clip:
                    out = np.clip(out, 0.0, 1.0)
                lines.append("{:.6f} {:.6f} {:.6f}".format(*out))

    # Write to file
    with open(filename, "w") as f:
        f.write("\n".join(lines))

    logger.log(logging.INFO, f"LUT written to {filename}")

def concatenate_color_transforms(A, B):
    """
    Concatenate two color transform matrices A and B.

    Parameters
    ----------
    A : np.ndarray
        Either 3x3 or 3x4 matrix.
    B : np.ndarray
        Either 3x3 or 3x4 matrix.

    Returns
    -------
    np.ndarray
        Combined 3x3 or 3x4 matrix equivalent to applying A then B.
    """
    if A.shape == (3, 4):
        A = np.vstack([A, [0, 0, 0, 1]])
    if B.shape == (3, 4):
        B = np.vstack([B, [0, 0, 0, 1]])
    result = B @ A
    return result[:3, :]

class VideoMarkerEditor(QtWidgets.QWidget):
    def __init__(self, videos, frame_number, csv_file):
        super().__init__()
        self.videos = videos
        self.frame_number = frame_number
        self.csv_file = csv_file
        self.labels = []  # store label names for colors
        self.colors = {}  # label -> color
        self.adjust_colors = False
        self.original_frames = {}  # store original frames for reset
        self.transformed_frames = {}  # store transformed frames for display
        self.color_transforms = {}

        self.init_ui()
        self.load_frames()
        if os.path.exists(self.csv_file):
            self.load_csv()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        # Video viewers in a grid
        self.viewers = {}
        self.scatter_items = {}
        self.viewer_layout = QtWidgets.QGridLayout()
        max_cols = 2
        for idx, (name, _) in enumerate(self.videos):
            pw = pg.GraphicsLayoutWidget()
            view = pw.addViewBox()
            view.setAspectLocked(True)
            view.invertY(True)
            img_item = pg.ImageItem()
            view.addItem(img_item)
            self.viewers[name] = {'view': view, 'img': img_item}
            scatter = pg.ScatterPlotItem(size=10)
            view.addItem(scatter)
            self.scatter_items[name] = scatter

            row = idx // max_cols
            col = idx % max_cols
            self.viewer_layout.addWidget(pw, row, col)
            pw.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

            # Mouse click
            pw.scene().sigMouseClicked.connect(lambda ev, n=name: self.on_click(ev, n))

        main_layout.addLayout(self.viewer_layout, stretch=1)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(1 + 2*len(self.videos))
        headers = ["label"]
        for name, _ in self.videos:
            headers += [f"{name}_x", f"{name}_y"]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setFixedHeight(200)
        self.table.itemChanged.connect(self.on_table_item_changed)
        self.table.currentCellChanged.connect(self.on_table_select)
        main_layout.addWidget(self.table, stretch=0)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Add marker")
        self.add_button.clicked.connect(self.add_marker_dialog)
        self.load_button = QtWidgets.QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_csv)
        self.save_button = QtWidgets.QPushButton("Save CSV")
        self.save_button.clicked.connect(self.save_csv)
        self.adjust_button = QtWidgets.QPushButton("Adjust color profile")
        self.adjust_button.setCheckable(True)
        self.adjust_button.toggled.connect(self.on_adjust_toggle)
        self.export_button = QtWidgets.QPushButton("Export LUTs")
        self.export_button.clicked.connect(self.export_luts)
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.adjust_button)
        button_layout.addWidget(self.export_button)
        main_layout.addLayout(button_layout)

    def export_luts(self, output_folder=None):
        if not self.color_transforms:
            QtWidgets.QMessageBox.warning(self, "No transforms", "Please apply color adjustment first.")
            return

        if output_folder is None:
            output_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder to save LUTs")
            if not output_folder:
                return

        os.makedirs(output_folder, exist_ok=True)
        for name, matrix in self.color_transforms.items():
            lut_file = os.path.join(output_folder, f"{name}.cube")
            # Ensure shape is 3x4
            if matrix.shape != (3, 4):
                QtWidgets.QMessageBox.warning(self, "Invalid transform", f"Video {name} transform is not 3x4.")
                continue
            export_matrix_as_cube(matrix, size=33, filename=lut_file)

    def load_frames(self, use_yuv=True):
        for name, filename in self.videos:
            if use_yuv:
                # --- Detect resolution ---
                ffprobe_cmd = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "json", filename
                ]
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
                info = json.loads(result.stdout)
                width = info["streams"][0]["width"]
                height = info["streams"][0]["height"]

                # --- Extract one raw YUV frame with ffmpeg ---
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-v", "error",
                    "-i", filename,
                    "-vf", f"select=eq(n\\,{self.frame_number})",
                    "-vframes", "1",
                    "-f", "rawvideo",
                    "-pix_fmt", "yuv420p",  # force known layout
                    "-"
                ]
                raw = subprocess.run(ffmpeg_cmd, capture_output=True).stdout

                # --- Parse YUV420p manually ---
                y_size = width * height
                uv_size = (width // 2) * (height // 2)
                y = np.frombuffer(raw[:y_size], dtype=np.uint8).reshape((height, width))
                u = np.frombuffer(raw[y_size:y_size + uv_size], dtype=np.uint8).reshape((height // 2, width // 2))
                v = np.frombuffer(raw[y_size + uv_size:], dtype=np.uint8).reshape((height // 2, width // 2))

                # Upsample U and V
                u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
                v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)

                yuv = np.stack([y, u, v], axis=-1).astype(np.float32, copy=False) / 255.0
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

            else:
                # --- Normal OpenCV read ---
                cap = cv2.VideoCapture(filename)
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    frame = np.zeros((480, 640, 3), np.float32)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32, copy=False) / 255.0

            self.original_frames[name] = frame
            self.transformed_frames[name] = frame.copy()
            self.viewers[name]['img'].setImage(np.clip((frame * 255), 0, 255).astype(np.uint8).transpose(1, 0, 2))

    def add_marker_dialog(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Add Marker", "Enter label name:")
        if ok and text:
            self.add_marker_row(text)

    def add_marker_row(self, label):
        if label in self.labels:
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
        self.labels.append(label)
        self.colors[label] = tuple(np.random.randint(50,255,3))
        for col in range(1, self.table.columnCount()):
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(""))
        self.update_markers()

    def on_click(self, event, video_name):
        if event.double():
            return
        pos = event.scenePos()
        vb = self.viewers[video_name]['view']
        img_coords = vb.mapSceneToView(pos)
        x = int(img_coords.x())
        y = int(img_coords.y())
        row = self.table.currentRow()
        if row == -1:
            return
        idx = [name for name,_ in self.videos].index(video_name)
        self.table.setItem(row, 1 + 2*idx, QtWidgets.QTableWidgetItem(str(x)))
        self.table.setItem(row, 2 + 2*idx, QtWidgets.QTableWidgetItem(str(y)))
        self.update_markers()

    def update_markers(self):
        current_row = self.table.currentRow()
        label_colors = self.get_label_colors(self.original_frames)
        for vid_idx, (name, _) in enumerate(self.videos):
            spots = []
            for row in range(self.table.rowCount()):
                label_item = self.table.item(row, 0)
                if not label_item:
                    continue
                label = label_item.text()
                x_item = self.table.item(row, 1 + 2 * vid_idx)
                y_item = self.table.item(row, 2 + 2 * vid_idx)
                if x_item and y_item and x_item.text() and y_item.text():
                    x = float(x_item.text())
                    y = float(y_item.text())
                    color = self.colors.get(label, (255, 0, 0))
                    image_color = color
                    if label in label_colors and name in label_colors[label]:
                        image_color = tuple(np.round(label_colors[label][name] * 255).astype(int))
                    brush = pg.mkBrush(*image_color)
                    pen = pg.mkPen(*color, width=4)
                    # Highlight selected row
                    #set border to label colors and filled area to color
                    if row == current_row:
                        spots.append({'pos': (x, y), 'brush': brush, 'pen': pen, 'size': 30, 'symbol': 't'})
                    else:
                        spots.append({'pos': (x, y), 'brush': brush, 'pen': pen, 'size': 15, 'symbol': 'o'})
            self.scatter_items[name].setData(spots)
        # If color adjustment is on, update images
        if self.adjust_colors:
            self.apply_color_adjustment()

    def on_table_item_changed(self, item):
        if item.column() == 0:
            old_label = None
            for lbl,row_idx in zip(self.labels, range(len(self.labels))):
                if row_idx == item.row():
                    old_label = lbl
                    break
            if old_label and old_label != item.text():
                self.colors[item.text()] = self.colors.pop(old_label)
                self.labels[item.row()] = item.text()
        self.update_markers()

    def on_table_select(self, currentRow, currentCol, prevRow, prevCol):
        self.update_markers()

    def save_csv(self):
        rows = []
        for row in range(self.table.rowCount()):
            row_dict = {}
            for col in range(self.table.columnCount()):
                header = self.table.horizontalHeaderItem(col).text()
                item = self.table.item(row, col)
                row_dict[header] = item.text() if item else ""
            rows.append(row_dict)
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        df.to_csv(self.csv_file, index=False)

    def load_csv(self):
        if not os.path.exists(self.csv_file):
            return
        df = pd.read_csv(self.csv_file)
        for _, row in df.iterrows():
            label = str(row['label'])
            if label not in self.labels:
                self.add_marker_row(label)
            for col in range(1, self.table.columnCount()):
                header = self.table.horizontalHeaderItem(col).text()
                val = str(row.get(header,""))
                self.table.setItem(self.labels.index(label), col, QtWidgets.QTableWidgetItem(val))
        self.update_markers()

    def on_adjust_toggle(self, checked):
        self.adjust_colors = checked
        if not checked:
            # restore original frames
            self.update_viewers(frames = self.original_frames)
        else:
            self.apply_color_adjustment()

    def videos_names(self):
        return [name for name,_ in self.videos]

    @staticmethod
    def average_hsv_color(colors:np.ndarray, weights=None, source_colorspace="rgb"):
        """
        Average a list of RGB colors (float32 in [0,1]) in a way that preserves
        perceived brightness and saturation using HSV space.
        """
        if len(colors) == 0:
            return np.zeros(3, dtype=np.float32)
        if source_colorspace == "yuv":
            colors = cv2.cvtColor(colors[np.newaxis, :, :].astype(np.float32, copy=False), cv2.COLOR_YUV2RGB)[0, :]

        # Convert to HSV (OpenCV expects float32 in [0,1])
        hsv = cv2.cvtColor(colors[np.newaxis, :, :].astype(np.float32, copy=False), cv2.COLOR_RGB2HSV)[0, :]

        # Hue in OpenCV is 0..179, convert to radians
        hues = np.deg2rad(hsv[:, 0])
        sats = hsv[:, 1]  # already in [0,1]
        vals = hsv[:, 2]  # already in [0,1]

        # Average hue on the unit circle
        x = np.cos(hues)
        y = np.sin(hues)
        mean_angle = np.arctan2(np.average(y, weights=weights), np.average(x, weights=weights))
        mean_h = np.rad2deg(mean_angle)  # back to 0..179
        if mean_h < 0:
            mean_h += 360.0

        # Average saturation and value
        mean_s = np.average(sats, weights=weights)
        mean_v = np.average(vals, weights=weights)

        # Recombine and convert back to RGB
        avg_hsv = np.array([mean_h, mean_s, mean_v], dtype=np.float32)[np.newaxis, np.newaxis, :]
        avg_rgb = cv2.cvtColor(avg_hsv, cv2.COLOR_HSV2RGB)[0, 0]
        if source_colorspace == "yuv":
            avg_rgb = cv2.cvtColor(avg_rgb[np.newaxis, np.newaxis,:], cv2.COLOR_RGB2YUV)[0,0]
        return avg_rgb

    def get_label_colors(self, images):
        label_colors = {}
        for row in range(self.table.rowCount()):
            label_item = self.table.item(row, 0)
            if not label_item:
                continue
            label = label_item.text()
            pts = {}
            for vid_idx, name in enumerate(self.videos_names()):
                x_item = self.table.item(row, 1 + 2 * vid_idx)
                y_item = self.table.item(row, 2 + 2 * vid_idx)
                if not x_item or not y_item or not x_item.text() or not y_item.text():
                    continue
                try:
                    x = float(x_item.text())
                    y = float(y_item.text())
                except ValueError:
                    continue
                if np.isnan(x) or np.isnan(y):
                    continue
                frame = images[name]
                h, w, _ = frame.shape
                xi = int(np.round(x))
                yi = int(np.round(y))
                xs = np.clip(np.arange(xi - 2, xi + 3), 0, w - 1)
                ys = np.clip(np.arange(yi - 2, yi + 3), 0, h - 1)
                patch = frame[np.ix_(ys, xs)]
                avg_color = patch.reshape(-1, 3).mean(axis=0)
                pts[name] = avg_color
            if len(pts) > 1:
                label_colors[label] = pts
        return label_colors

    def apply_color_adjustment(self, iterations=5, gamma=1, colorspace='rgb', affine=True):
        """Iteratively compute per-video linear or affine transforms to minimize color difference at markers.
           Supports RGB or YUV color spaces and optional gamma correction.
        """
        if not self.labels:
            return

        # Convert all original frames to float32 once
        transformed_float = {name: self.original_frames[name].astype(np.float32, copy=False) for name in self.videos_names()}

        # Apply gamma linearization if needed
        if gamma != 1.0:
            for name in transformed_float:
                transformed_float[name] = transformed_float[name] ** gamma

        # Convert to YUV if requested
        if colorspace.lower() == 'yuv':
            for name in transformed_float:
                frame = transformed_float[name]
                transformed_float[name] = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

        np.set_printoptions(precision=3, suppress=True)
        full_color_transforms = {}

        for it in range(iterations):
            label_colors = self.get_label_colors(transformed_float)  # label -> list of (video_name, color)
            if not label_colors:
                return
            color_data = []
            for label, pts in label_colors.items():
                row = {'label': label}
                for name, color in pts.items():
                    row[f'{name}_color'] = np.round(color, 3)  # keep 3 decimals
                color_data.append(row)
            df_colors = pd.DataFrame(color_data)
            logger.info(f"Iteration {it + 1}, average colors at markers:\n{df_colors}")

            # Solve transform for each video
            color_transforms = {}
            lambda_reg = 0.01  # regularization strength, adjust as needed
            average_colors = {}
            for label, pts in label_colors.items():
                ref_colors, weights = [], []
                for n, c in pts.items():
                    ref_colors.append(c)
                    weights.append(args.weights[self.videos_names().index(n)] if args.weights else 1.0)
                ref_colors = np.array(ref_colors)  # Kx3
                # target = np.average(ref_colors, axis=0, weights=weights)
                average_colors[label] = VideoMarkerEditor.average_hsv_color(ref_colors, weights=weights, source_colorspace=colorspace)
            for name in self.videos_names():
                X, Y = [], []
                for label, pts in label_colors.items():
                    selected_color = pts.get(name, None)
                    if selected_color is None:
                        continue
                    if affine:
                        X.append(np.append(selected_color, 1))
                    else:
                        X.append(selected_color)
                    Y.append(average_colors[label])
                if not X:
                    # fallback identity
                    color_transforms[name] = np.eye(3, 4 if affine else 3, dtype=np.float32)
                    continue

                X = np.array(X)  # Nx4 (affine) or Nx3 (linear)
                Y = np.array(Y)  # Nx3

                if affine:
                    # X: Nx4, Y: Nx3, A: 4x3
                    A_target = np.eye(4, 3, dtype=np.float32)  # 4x3 identity-ish
                else:
                    # X: Nx3, Y: Nx3, A: 3x3
                    A_target = np.eye(3, dtype=np.float32)  # 3x3

                # Regularized least squares: (X^T X + λ I) A = X^T Y + λ A_target
                XtX = X.T @ X
                reg = lambda_reg * np.eye(X.shape[1], dtype=np.float32)  # shape matches XtX
                A = np.linalg.solve(XtX + reg, X.T @ Y + lambda_reg * A_target)
                color_transforms[name] = A.T.astype(np.float32)
                if name not in full_color_transforms:
                    full_color_transforms[name] = color_transforms[name]
                else:
                    full_color_transforms[name] = concatenate_color_transforms(full_color_transforms[name], color_transforms[name])

                logger.log(logging.INFO,
                    f"Iteration {it + 1}, Video '{name}' transform matrix (regularized):\n{color_transforms[name]}")
                #print marker with biggest error
                errors = np.linalg.norm((X @ color_transforms[name].T) - Y, axis=1)
                max_err_idx = np.argmax(errors)
                logger.log(logging.INFO, f"  Max error for video {name} is label '{list(label_colors.keys())[max_err_idx]}': {errors[max_err_idx]:.4f}")

            # Apply transform
            for name in self.videos_names():
                frame = transformed_float[name]
                h, w, _ = frame.shape
                flat = frame.reshape(-1, 3)
                flat = flat @ color_transforms[name].T[:3, :3]
                if affine:
                    flat = flat + color_transforms[name].T[3, :]
                transformed_float[name] = flat.reshape(h, w, 3)

        self.color_transforms = full_color_transforms

        # Convert back to RGB if needed
        if colorspace.lower() == 'yuv':
            for name in transformed_float:
                transformed_float[name] = cv2.cvtColor(transformed_float[name].astype(np.float32, copy=False), cv2.COLOR_YUV2RGB)

        # Apply inverse gamma
        if gamma != 1.0:
            for name in transformed_float:
                frame = transformed_float[name]
                transformed_float[name] = frame ** (1.0 / gamma)
        for name in transformed_float:
            self.transformed_frames[name] = transformed_float[name]
        self.update_viewers(frames=transformed_float)

    def update_viewers(self, frames):
        # Update display
        for name in self.videos_names():
            self.viewers[name]['img'].setImage(np.clip(frames[name] * 255, 0, 255).astype(np.uint8).transpose(1, 0, 2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs='+', help="Video name and filename pairs", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--weights", default=None, nargs='+', type=float)
    parser.add_argument("--csv", type=str, default="markers.csv")
    parser.add_argument("--no-gui", action='store_true', help="Run without GUI")
    parser.add_argument("--output-original", type=str, default=None,
                        help="Output preview image of adjusted frames at startup")
    parser.add_argument("--output-preview", type=str, default=None, help="Output preview image of adjusted frames at startup")
    parser.add_argument("--output-luts", type=str, default=None, help="Automatically export LUTs after adjustment at startup")
    parser.add_argument("--loglevel", type=str, default="INFO", help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is INFO.")
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if len(args.videos)%2 != 0:
        print("Error: videos must be given as name filename pairs")
        sys.exit(1)
    videos = [(args.videos[i], args.videos[i+1]) for i in range(0,len(args.videos),2)]

    app = QtWidgets.QApplication(sys.argv)
    editor = VideoMarkerEditor(videos, args.frame, args.csv)
    editor.setWindowTitle("Manual Marker Editor (Color Adjustable)")
    if args.output_luts is not None:
        editor.apply_color_adjustment()
        editor.export_luts(output_folder=os.path.dirname(args.output_luts))
    if args.output_preview is not None:
        editor.apply_color_adjustment()
        scaled_previews = []
        # scale all videos have the height 1080
        for name, _ in videos:
            frame = editor.transformed_frames[name]
            h, w, _ = frame.shape
            if h != 1080:
                scale = 1080 / h
                new_w = int(w * scale)
                frame = cv2.resize(frame, (new_w, 1080), interpolation=cv2.INTER_LINEAR)
            scaled_previews.append(frame)
        if args.output_preview.endswith(".png"):
            combined = np.hstack(scaled_previews)
            cv2.imwrite(args.output_preview, np.clip(combined * 255, 0, 255).astype(np.uint8)[:,:,::-1])
        else:
            os.makedirs(args.output_preview, exist_ok=True)
            for i,(name,_) in enumerate(videos):
                frame = scaled_previews[i]
                cv2.imwrite(os.path.join(args.output_preview, f"{name}_preview.png"), np.clip(frame * 255, 0, 255).astype(np.uint8)[:,:,::-1])
    if args.output_original is not None:
        scaled_previews = []
        #scale all videos have the height 1080
        for name,_ in videos:
            frame = editor.original_frames[name]
            h, w, _ = frame.shape
            if h != 1080:
                scale = 1080 / h
                new_w = int(w * scale)
                frame = cv2.resize(frame, (new_w, 1080), interpolation=cv2.INTER_LINEAR)
            scaled_previews.append(frame)

        if args.output_original.endswith(".png"):
            combined = np.hstack(scaled_previews)
            cv2.imwrite(args.output_original, np.clip(combined * 255, 0, 255).astype(np.uint8)[:,:,::-1])
        else:
            os.makedirs(args.output_original, exist_ok=True)
            for i,(name,_) in enumerate(videos):
                frame = scaled_previews[i]
                cv2.imwrite(os.path.join(args.output_original, f"{name}_original.png"), np.clip(frame * 255, 0, 255).astype(np.uint8)[:,:,::-1])
                logger.log(logging.INFO, f"Wrote original preview to {os.path.join(args.output_original, f'{name}_original.png')}")
    if not args.no_gui:
        editor.show()
        sys.exit(app.exec())
