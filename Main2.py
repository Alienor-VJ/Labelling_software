import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import math


class ImageSegmentationApp:
    def __init__(self, root):
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
        self.selected_point = None  # The selected point in the shape to modify
        self.categories = {"Sure": [], "Unsure": [], "Plaque": []}

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

        # Adding titles for the lists
        self.title_sure = Label(self.list_frame, text="Sure", font=("Helvetica", 14, "bold"))
        self.title_sure.pack()
        self.table_sure = Text(self.list_frame, height=10, width=40)
        self.table_sure.pack(fill=Y)

        self.title_unsure = Label(self.list_frame, text="Unsure", font=("Helvetica", 14, "bold"))
        self.title_unsure.pack()
        self.table_unsure = Text(self.list_frame, height=10, width=40)
        self.table_unsure.pack(fill=Y)

        self.title_plaque = Label(self.list_frame, text="Plaque", font=("Helvetica", 14, "bold"))
        self.title_plaque.pack()
        self.table_plaque = Text(self.list_frame, height=10, width=40)
        self.table_plaque.pack(fill=Y)

        # Adding mean area labels
        self.mean_area_label_sure = Label(self.list_frame, text="Mean Area (Sure): -", font=("Helvetica", 12))
        self.mean_area_label_sure.pack()
        self.mean_area_label_unsure = Label(self.list_frame, text="Mean Area (Unsure): -", font=("Helvetica", 12))
        self.mean_area_label_unsure.pack()
        self.mean_area_label_plaque = Label(self.list_frame, text="Mean Area (Plaque): -", font=("Helvetica", 12))
        self.mean_area_label_plaque.pack()

        # Bind events
        self.canvas.bind("<Button-1>", self.click_event)
        self.canvas.bind("<B1-Motion>", self.drag_event)
        self.canvas.bind("<ButtonRelease-1>", self.release_event)
        self.root.bind("<Return>", self.save_and_close)
        self.root.bind("<s>", self.suppress_shape)  # Bind the 'S' key to suppress the shape
        self.root.bind("<a>", self.deselect_shape)  # Bind the 'A' key to deselect the shape
        self.root.bind("<u>", self.set_unsure)  # Bind the 'U' key to categorize as unsure
        self.root.bind("<p>", self.set_plaque)  # Bind the 'P' key to categorize as plaque
        self.root.bind("<Escape>", self.close_app)

    def compute_distance(self, point):
        # Compute the Euclidean distance from the center of the image
        center = (self.original_img.shape[1] // 2, self.original_img.shape[0] // 2)
        return math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)

    def compute_ellipse_ratio(self, points):
        # Fit an ellipse and compute the ratio of the larger to smaller axis
        if len(points) < 5:  # Ellipse fitting requires at least 5 points
            return None

        # Fit ellipse
        ellipse = cv2.fitEllipse(np.array(points, dtype=np.float32))
        (center, (major_axis, minor_axis), angle) = ellipse
        if minor_axis != 0:
            return major_axis / minor_axis
        return None

    def update_area_tables(self):
        # Clear previous content in the tables
        self.table_sure.delete(1.0, END)
        self.table_unsure.delete(1.0, END)
        self.table_plaque.delete(1.0, END)

        # Store the areas for each category to calculate the mean area
        sure_areas = []
        unsure_areas = []
        plaque_areas = []

        for category, shapes in self.categories.items():
            for shape in shapes:
                area = self.compute_area(shape)
                distance = self.compute_distance(shape[0])  # Take the first point for simplicity
                ellipse_ratio = self.compute_ellipse_ratio(shape)

                # Display the values in the correct table
                table_entry = f"Area: {area:.2f}, Dist: {distance:.2f}, Ellipse Ratio: {ellipse_ratio:.2f if ellipse_ratio else 'N/A'}\n"

                if category == "Sure":
                    self.table_sure.insert(END, table_entry)
                    sure_areas.append(area)
                elif category == "Unsure":
                    self.table_unsure.insert(END, table_entry)
                    unsure_areas.append(area)
                elif category == "Plaque":
                    self.table_plaque.insert(END, table_entry)
                    plaque_areas.append(area)

        # Calculate and update the mean area for each category
        self.update_mean_area(sure_areas, unsure_areas, plaque_areas)

    def update_mean_area(self, sure_areas, unsure_areas, plaque_areas):
        # Calculate the mean area for each category and update the labels
        mean_sure = sum(sure_areas) / len(sure_areas) if sure_areas else 0
        mean_unsure = sum(unsure_areas) / len(unsure_areas) if unsure_areas else 0
        mean_plaque = sum(plaque_areas) / len(plaque_areas) if plaque_areas else 0

        # Update the mean area labels
        self.mean_area_label_sure.config(text=f"Mean Area (Sure): {mean_sure:.2f}")
        self.mean_area_label_unsure.config(text=f"Mean Area (Unsure): {mean_unsure:.2f}")
        self.mean_area_label_plaque.config(text=f"Mean Area (Plaque): {mean_plaque:.2f}")

    def compute_area(self, shape):
        # Compute the area of the shape (polygon)
        return cv2.contourArea(np.array(shape))

    def save_and_close(self, event=None):
        # Save the current shapes into a file or database (e.g., CSV, JSON)
        with open("shapes_data.txt", "w") as file:
            for category, shapes in self.categories.items():
                for shape in shapes:
                    file.write(f"{category}: {shape}\n")

        print("Shapes saved successfully!")
        self.root.quit()  # Close the application

    def close_app(self, event=None):
        # Close the application without saving
        self.root.quit()

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
        if self.image is None:
            print("Error: No image loaded!")
            return

        temp_image = self.image.copy()

        # Draw shapes on the image
        for i, (shape, category) in enumerate(self.categories.items()):
            for s in category:
                pts = np.array(s, dtype=np.int32)
                if len(pts) < 3:
                    continue

                # Compute the area of the polygon
                area = cv2.contourArea(pts)
                print(f"Shape {i + 1} area: {area:.2f}")

                # Draw the shape on the image
                pts_contour = pts.reshape((-1, 1, 2))
                color = (0, 255, 0)  # Default color for sure category
                if category == 'Unsure':
                    color = (0, 255, 255)  # Yellow for unsure
                elif category == 'Plaque':
                    color = (255, 0, 0)  # Blue for plaque
                elif self.selected_shape == s:
                    color = (255, 0, 0)  # Blue when selected for modification
                cv2.polylines(temp_image, [pts_contour], isClosed=True, color=color, thickness=2)

        # Draw the current drawing (if any)
        if self.drawing and len(self.current_shape) > 1:
            pts_contour = np.array(self.current_shape, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_image, [pts_contour], isClosed=False, color=(0, 255, 255), thickness=2)

        # Convert image for display
        image_rgb = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(image_rgb))

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=NW, image=img)
        self.canvas.image = img

        # Update the area tables for each category
        self.update_area_tables()

    def update_area_tables(self):
        # Update the area tables for each category
        self.table_sure.delete("1.0", END)
        self.table_unsure.delete("1.0", END)
        self.table_plaque.delete("1.0", END)

        for category, shapes in self.categories.items():
            total_area = 0
            for shape in shapes:
                pts = np.array(shape, dtype=np.int32)
                if len(pts) < 3:
                    continue
                area = cv2.contourArea(pts)
                total_area += area
                if category == "Sure":
                    self.table_sure.insert(END, f"Shape: {area:.2f} pixels²\n")
                elif category == "Unsure":
                    self.table_unsure.insert(END, f"Shape: {area:.2f} pixels²\n")
                elif category == "Plaque":
                    self.table_plaque.insert(END, f"Shape: {area:.2f} pixels²\n")

            if category == "Sure":
                self.table_sure.insert(END, f"Total Area: {total_area:.2f} pixels²")
            elif category == "Unsure":
                self.table_unsure.insert(END, f"Total Area: {total_area:.2f} pixels²")
            elif category == "Plaque":
                self.table_plaque.insert(END, f"Total Area: {total_area:.2f} pixels²")

    def click_event(self, event):
        # If the user clicks near the border of a shape, select it
        for shape in self.categories["Sure"] + self.categories["Unsure"] + self.categories["Plaque"]:
            for j, point in enumerate(shape):
                dx = event.x - point[0]
                dy = event.y - point[1]
                if abs(dx) < 10 and abs(dy) < 10:  # Select point if close enough
                    self.selected_shape = shape
                    self.selected_point = j
                    self.show_image()
                    return

        # Start drawing a new shape if no selection
        if self.selected_shape is None:
            self.current_shape = [(event.x, event.y)]
            self.drawing = True

    def drag_event(self, event):
        # If we are modifying a selected point, move it
        if self.selected_shape is not None and self.selected_point is not None:
            self.selected_shape[self.selected_point] = (event.x, event.y)
            self.show_image()

        # If drawing a new shape, keep adding points and redraw the shape
        if self.drawing:
            self.current_shape.append((event.x, event.y))
            self.show_image()

    def release_event(self, event):
        # If we are drawing a shape, finish it when the mouse is released
        if self.drawing:
            self.shapes.append(self.current_shape)
            self.categories["Sure"].append(self.current_shape)
            self.drawing = False
            self.show_image()

    def suppress_shape(self, event):
        # Suppress the selected shape
        if self.selected_shape is not None:
            for category in self.categories.values():
                if self.selected_shape in category:
                    category.remove(self.selected_shape)
                    break
            self.selected_shape = None  # Clear the selection
            self.show_image()  # Refresh the image

    def deselect_shape(self, event):
        # Deselect the current shape
        self.selected_shape = None
        self.selected_point = None
        self.show_image()  # Refresh the image

    def set_unsure(self, event):
        if self.selected_shape:
            self.move_shape_to_category("Unsure")
            self.show_image()

    def set_plaque(self, event):
        if self.selected_shape:
            self.move_shape_to_category("Plaque")
            self.show_image()

    def move_shape_to_category(self, category):
        # Remove shape from all categories and add to the selected category
        for cat in self.categories:
            if self.selected_shape in self.categories[cat]:
                self.categories[cat].remove(self.selected_shape)
                break
        self.categories[category].append(self.selected_shape)

    def close_app(self, event):
        self.root.destroy()  # This will close the app

if __name__ == "__main__":
    root = Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()