import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import math

class ImageSegmentationApp:
    def __init__(self, root):
        print("Initializing App")
        self.root = root
        self.root.title("Segmentation and Area Calculation")
        self.root.attributes("-fullscreen", True)

        self.img_path = None
        self.image = None
        self.original_img = None
        self.mask = None
        self.drawing = False
        self.current_shape = []  # List of points (vertices) of the current polygon
        self.shapes = []  # List of all drawn shapes
        self.selected_shape = None
        self.selected_points = []  # List of points selected for modification
        self.proposed_reshape = None  # Store the proposed reshaped shape
        self.spot_data = []

        self.load_button = Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        # Create a Frame to arrange the image and the list of shapes side by side
        self.main_frame = Frame(root)
        self.main_frame.pack(fill=BOTH, expand=True)

        # Create a Canvas to display the image
        self.canvas_frame = Frame(self.main_frame)
        self.canvas_frame.pack(side=LEFT, fill=BOTH, expand=True)

        self.canvas = Canvas(self.canvas_frame)
        self.canvas.pack(fill=BOTH, expand=True)

        # Create a Frame for the list of shapes (on the right side)
        self.list_frame = Frame(self.main_frame, width=300)
        self.list_frame.pack(side=RIGHT, fill=Y)

        self.table = Text(self.list_frame, height=30, width=40)  # Adjusted size for the surface area list
        self.table.pack(fill=Y)

        # Bind events
        self.canvas.bind("<Button-1>", self.click_event)
        self.canvas.bind("<B1-Motion>", self.drag_event)
        self.canvas.bind("<ButtonRelease-1>", self.release_event)
        self.root.bind("<Return>", self.save_and_close)
        self.root.bind("<s>", self.suppress_shape)  # Bind the 'S' key to suppress the shape
        self.root.bind("<a>", self.deselect_shape)  # Bind the 'A' key to deselect the shape
        self.root.bind("<d>", self.accept_reshape)  # Bind the 'D' key to accept the reshape
        self.root.bind("<f>", self.reject_reshape)  # Bind the 'F' key to reject the reshape

        self.root.bind("<Escape>", self.close_app)

    def load_image(self):
        self.img_path = filedialog.askopenfilename()
        if self.img_path:
            self.original_img = cv2.imread(self.img_path)
            if self.original_img is None:
                print("Error loading image!")
                return

            # Resize the image to fit within the screen while maintaining aspect ratio
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Get the image's original dimensions
            img_height, img_width = self.original_img.shape[:2]

            # Calculate scale factor to fit the image to the screen size
            scale_factor = min(screen_width / img_width, screen_height / img_height)

            # Calculate new dimensions
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)

            # Resize the image to fit
            self.image = cv2.resize(self.original_img, (new_width, new_height))
            self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

            # Resize the canvas to match the resized image
            self.canvas.config(width=new_width, height=new_height)

            self.show_image()

    def show_image(self):
        temp_image = self.image.copy()

        # Draw the shapes already created
        for i, shape in enumerate(self.shapes):
            if len(shape) > 1:
                color = (255, 255, 0) if i == self.selected_shape else (0, 255, 0)
                cv2.polylines(temp_image, [np.array(shape)], isClosed=True, color=color, thickness=2)

        # Draw the proposed reshape (in red)
        if self.proposed_reshape is not None:
            for reshape in self.proposed_reshape:
                cv2.polylines(temp_image, [np.array(reshape)], isClosed=True, color=(0, 0, 255), thickness=2)

        # Draw the shape being drawn (real-time)
        if self.drawing and len(self.current_shape) > 1:
            cv2.polylines(temp_image, [np.array(self.current_shape)], isClosed=False, color=(0, 255, 255), thickness=2)

        image_rgb = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(image_rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=NW, image=img)
        self.canvas.image = img

        # Update the area table
        self.table.delete("1.0", END)
        total_area = 0
        areas = []

        for i, shape in enumerate(self.shapes):
            if len(shape) > 2:
                area = cv2.contourArea(np.array(shape))
                areas.append(area)
                total_area += area
                self.table.insert(END, f"Shape {i + 1}: {area:.2f} pixels²\n")

        if areas:
            mean_area = total_area / len(areas)
            self.table.insert(END, f"\nTotal Surface Area: {total_area:.2f} pixels²")
            self.table.insert(END, f"\nMean Surface Area: {mean_area:.2f} pixels²")
        else:
            self.table.insert(END, f"\nTotal Surface Area: 0.00 pixels²")
            self.table.insert(END, f"\nMean Surface Area: 0.00 pixels²")

    def click_event(self, event):
        # If the user clicks near the border of a shape, select it
        for i, shape in enumerate(self.shapes):
            for j, point in enumerate(shape):
                dx = event.x - point[0]
                dy = event.y - point[1]
                if abs(dx) < 10 and abs(dy) < 10:  # Select point if close enough
                    self.selected_shape = i
                    self.selected_points = self.get_nearby_points(i, event.x,
                                                                  event.y)  # Select nearby points for modification
                    self.show_image()
                    return

        # Start drawing a new shape if no selection
        if self.selected_shape is None:
            self.current_shape = [(event.x, event.y)]
            self.drawing = True

    def drag_event(self, event):
        # If a shape is selected, move the selected points
        if self.selected_shape is not None and self.selected_points:
            for idx in self.selected_points:
                self.shapes[self.selected_shape][idx] = (event.x, event.y)
            self.show_image()
            return

        # If drawing a new shape, keep adding points as the user drags the mouse
        if self.drawing:
            self.current_shape.append((event.x, event.y))
            self.show_image()

    def release_event(self, event):
        # If we are drawing a shape, finish it when the mouse is released
        if self.drawing:
            self.shapes.append(self.current_shape)
            self.drawing = False

            # Propose a reshape for the newly drawn shape
            self.propose_reshape(self.current_shape)

            self.show_image()

    def propose_reshape(self, shape):
        # Propose a more sophisticated reshape
        expanded = self.expand_shape(shape)
        self.proposed_reshape = expanded

    def expand_shape(self, shape):
        contour = np.array(shape, dtype=np.int32)

        # Canny edge detection for boundary
        edges = cv2.Canny(self.mask, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        expanded_contour = []
        for cnt in contours:
            if cv2.pointPolygonTest(cnt, (shape[0][0], shape[0][1]), False) >= 0:
                expanded_contour.append(cnt)

        # Apply smoothing (optional), here we use a basic Gaussian blur
        smoothed_contour = cv2.GaussianBlur(np.array(expanded_contour, dtype=np.float32), (5, 5), 0)

        return smoothed_contour

    def accept_reshape(self, event):
        # Accept the proposed reshape and add it to the shapes
        if self.proposed_reshape is not None:
            self.shapes.append(self.proposed_reshape)
            self.proposed_reshape = None  # Clear the proposed reshape
            self.show_image()

    def reject_reshape(self, event):
        # Reject the proposed reshape
        self.proposed_reshape = None  # Clear the proposed reshape
        self.show_image()

    def save_and_close(self, event):
        # Save the current image with annotations
        if self.img_path:
            save_path = self.img_path.replace(".jpg", "_annotated.jpg").replace(".png", "_annotated.png")
            temp_image = self.original_img.copy()
            for shape in self.shapes:
                if len(shape) > 1:  # Only draw shapes with more than one point
                    cv2.polylines(temp_image, [np.array(shape)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imwrite(save_path, temp_image)
            print(f"Image saved as {save_path}")

        # Reset for a new image load
        self.reset_app()

        # Reopen the image file dialog to load a new image
        self.load_image()

    def reset_app(self):
        # Reset the current state for a new image
        self.shapes = []  # Clear all shapes
        self.selected_shape = None
        self.selected_points = []
        self.current_shape = []
        self.drawing = False
        self.table.delete("1.0", END)  # Clear the table
        self.canvas.delete("all")  # Clear the canvas

    def suppress_shape(self, event):
        # Suppress the selected shape
        if self.selected_shape is not None:
            del self.shapes[self.selected_shape]  # Remove the selected shape
            self.selected_shape = None  # Clear the selection
            self.show_image()  # Refresh the image

    def deselect_shape(self, event):
        # Deselect the current shape
        self.selected_shape = None
        self.selected_points = []
        self.show_image()  # Refresh the image

    def get_nearby_points(self, shape_idx, x, y):
        # Find nearby points in the selected shape to make modification smoother
        shape = self.shapes[shape_idx]
        selected_points = []
        for i, point in enumerate(shape):
            dx = x - point[0]
            dy = y - point[1]
            if abs(dx) < 15 and abs(dy) < 15:
                selected_points.append(i)  # Select the point
        return selected_points

    def close_app(self, event):
        self.root.quit()  # Close the application


if __name__ == "__main__":
    root = Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
