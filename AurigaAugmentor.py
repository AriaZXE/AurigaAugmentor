import os
import datetime
import random
import shutil
import threading
from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageChops
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.core.image import Image as CoreImage
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, DictProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

# Set the window size
Window.size = (1200, 800)

# ------------------ Loading Screen ------------------
class LoadingScreen(Screen):
    loading_text = StringProperty("Loading")
    counter = NumericProperty(0)
    event = None

    def on_enter(self):
        self.counter = 0
        self.loading_text = "Loading"
        # Update the loading text every 0.5 seconds
        self.event = Clock.schedule_interval(self.update_loading, 0.5)
        # Keep the loading screen visible for 5 seconds
        Clock.schedule_once(self.switch_to_folder_screen, 5)

    def update_loading(self, dt):
        self.counter = (self.counter + 1) % 4
        self.loading_text = "Loading" + "." * self.counter

    def switch_to_folder_screen(self, dt):
        if self.event:
            self.event.cancel()
        self.manager.current = 'folder_selection'

# ------------------ Folder Selection Screen ------------------
class FolderSelectionScreen(Screen):
    image_folder = StringProperty("")
    label_folder = StringProperty("")
    image_count = NumericProperty(0)
    label_count = NumericProperty(0)

    def validate_folders(self):
        images_exist = False
        labels_exist = False
        if os.path.isdir(self.image_folder):
            images = [fname for fname in os.listdir(self.image_folder)
                      if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            images_exist = len(images) > 0
            self.image_count = len(images)
        if os.path.isdir(self.label_folder):
            labels = [fname for fname in os.listdir(self.label_folder)
                      if fname.lower().endswith('.txt')]
            labels_exist = len(labels) > 0
            self.label_count = len(labels)
        if not images_exist or not labels_exist:
            self.ids.warning_label.text = "Warning: Folders must contain valid image and text files."
        else:
            self.ids.warning_label.text = ""
            sample_screen = self.manager.get_screen('sample')
            sample_screen.load_first_image(self.image_folder)
            sample_screen.update_total_augmented()
            self.manager.current = 'sample'

    def browse_folder(self, folder_type):
        content = BoxLayout(orientation='vertical')
        from kivy.uix.filechooser import FileChooserListView
        # For image folder, start at the user's home directory instead of root
        if folder_type == 'image':
            start_path = os.path.expanduser("~")
        elif folder_type == 'label' and self.image_folder:
            start_path = os.path.dirname(self.image_folder)
        else:
            start_path = os.path.expanduser("~")
        filechooser = FileChooserListView(path=start_path, dirselect=True)
        content.add_widget(filechooser)
        select_button = Button(text="Select", size_hint_y=None, height='40dp')
        content.add_widget(select_button)
        popup = Popup(title="Select Folder", content=content, size_hint=(0.9, 0.9))
        select_button.bind(on_release=lambda x: self._set_folder(filechooser, folder_type, popup))
        popup.open()

    def _set_folder(self, filechooser, folder_type, popup):
        selected_path = filechooser.selection[0] if filechooser.selection else filechooser.path
        if folder_type == 'image':
            self.image_folder = selected_path
            self.ids.image_folder_input.text = selected_path
            try:
                self.image_count = len([f for f in os.listdir(selected_path)
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            except Exception:
                self.image_count = 0
            self.ids.image_count_label.text = f"Image files: {self.image_count}"
            # If a subfolder named "labels" exists in the image folder, auto-select it
            candidate = os.path.join(selected_path, "labels")
            if os.path.isdir(candidate):
                self.label_folder = candidate
                self.ids.label_folder_input.text = candidate
                try:
                    self.label_count = len([f for f in os.listdir(candidate)
                                            if f.lower().endswith('.txt')])
                except Exception:
                    self.label_count = 0
                self.ids.label_count_label.text = f"Label files: {self.label_count}"
        elif folder_type == 'label':
            self.label_folder = selected_path
            self.ids.label_folder_input.text = selected_path
            try:
                self.label_count = len([f for f in os.listdir(selected_path)
                                        if f.lower().endswith('.txt')])
            except Exception:
                self.label_count = 0
            self.ids.label_count_label.text = f"Label files: {self.label_count}"
        popup.dismiss()

# ------------------ Sample Screen ------------------
class SampleScreen(Screen):
    image_folder = StringProperty("")
    current_index = NumericProperty(0)
    current_filter = StringProperty("Salt and Pepper")
    filter_settings = DictProperty({})

    def __init__(self, **kwargs):
        super(SampleScreen, self).__init__(**kwargs)
        filters = ["Salt and Pepper", "Brightness Increase", "Brightness Decrease",
                   "Saturation Increase", "Blur", "Sunlight", "Shake Blur", "Shadow", "Hue"]
        self.filter_settings = {f: {"count": 1, "intensity": 50} for f in filters}

    def on_pre_enter(self):
        self.ids.filter_spinner.text = self.current_filter

    def load_first_image(self, folder):
        self.image_folder = folder
        self.current_index = 0
        self.display_image()

    def display_image(self):
        image_files = [f for f in os.listdir(self.image_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if image_files:
            path = os.path.join(self.image_folder, image_files[self.current_index])
            self.ids.sample_image.source = path
            self.update_preview()

    def next_image(self):
        image_files = [f for f in os.listdir(self.image_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if image_files:
            self.current_index = (self.current_index + 1) % len(image_files)
            self.display_image()

    def prev_image(self):
        image_files = [f for f in os.listdir(self.image_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if image_files:
            self.current_index = (self.current_index - 1) % len(image_files)
            self.display_image()

    def on_filter_change(self, new_filter):
        # Save current filter settings before switching
        self.save_current_filter_settings()
        self.current_filter = new_filter
        # Retrieve saved intensity for the new filter, or use default if not set
        intensity = self.filter_settings.get(new_filter, {}).get("intensity")
        if intensity is None:
            intensity = self.ids.filter_slider.value
            self.filter_settings[new_filter] = {
                "count": int(self.ids.filter_count.text),
                "intensity": intensity
            }
        # Update the slider and count input with the saved values
        self.ids.filter_slider.value = intensity
        self.ids.filter_count.text = str(self.filter_settings[new_filter]["count"])
        self.update_preview()
        self.update_total_augmented()

    def on_slider_change(self, value):
        self.filter_settings[self.current_filter]["intensity"] = int(value)
        self.update_preview()

    def save_current_filter_settings(self):
        try:
            count = int(self.ids.filter_count.text)
        except Exception:
            count = 1
        intensity = int(self.ids.filter_slider.value)
        self.filter_settings[self.current_filter] = {"count": count, "intensity": intensity}
        self.update_total_augmented()

    def next_filter(self):
        filters = ["Salt and Pepper", "Brightness Increase", "Brightness Decrease",
                   "Saturation Increase", "Blur", "Sunlight", "Shake Blur", "Shadow", "Hue"]
        current = filters.index(self.current_filter)
        next_index = (current + 1) % len(filters)
        self.ids.filter_spinner.text = filters[next_index]
        self.on_filter_change(filters[next_index])

    def update_preview(self):
        source_path = self.ids.sample_image.source
        if not source_path or not os.path.exists(source_path):
            return
        try:
            img = Image.open(source_path).convert("RGB")
        except Exception as e:
            print("Error opening image:", e)
            return
        intensity = self.filter_settings[self.current_filter]["intensity"]
        if self.current_filter == "Salt and Pepper":
            np_img = np.array(img)
            row, col, ch = np_img.shape
            s_vs_p = 0.5
            amount = intensity / 500.0
            num_salt = np.ceil(amount * np_img.size * s_vs_p)
            num_pepper = np.ceil(amount * np_img.size * (1.0 - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in np_img.shape[:2]]
            np_img[coords[0], coords[1], :] = 255
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in np_img.shape[:2]]
            np_img[coords[0], coords[1], :] = 0
            img = Image.fromarray(np.uint8(np_img))
        elif self.current_filter == "Brightness Increase":
            enhancer = ImageEnhance.Brightness(img)
            factor = 1 + intensity/100.0
            img = enhancer.enhance(factor)
        elif self.current_filter == "Brightness Decrease":
            enhancer = ImageEnhance.Brightness(img)
            factor = max(0.1, 1 - intensity/100.0)
            img = enhancer.enhance(factor)
        elif self.current_filter == "Saturation Increase":
            enhancer = ImageEnhance.Color(img)
            factor = 1 + intensity/100.0
            img = enhancer.enhance(factor)
        elif self.current_filter == "Blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=intensity/10.0))
        elif self.current_filter == "Sunlight":
            width, height = img.size
            points = [(random.randint(0, width-1), random.randint(0, height-1)) for _ in range(3)]
            mask = Image.new("L", img.size, 0)
            d = ImageDraw.Draw(mask)
            d.polygon(points, fill=255)
            enhancer = ImageEnhance.Brightness(img)
            bright_img = enhancer.enhance(1 + intensity/100.0)
            img = Image.composite(bright_img, img, mask)
        elif self.current_filter == "Shake Blur":
            # Use a separate variable for the count loop (j) to avoid conflicts
            # and use 'k' for internal iterations.
            # The filter applies a random offset multiple times and averages the result.
            pass  # Shake Blur processing is done in augmentation.
        elif self.current_filter == "Shadow":
            img_rgba = img.convert("RGBA")
            overlay = Image.new("RGBA", img_rgba.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            num_polygons = random.randint(3, 5)
            for _ in range(num_polygons):
                num_vertices = random.randint(3, 6)
                points = [(random.randint(0, img_rgba.size[0]-1), random.randint(0, img_rgba.size[1]-1)) for _ in range(num_vertices)]
                alpha = int(50 + intensity/2)
                if alpha > 200:
                    alpha = 200
                draw.polygon(points, fill=(0, 0, 0, alpha))
            img = Image.alpha_composite(img_rgba, overlay).convert("RGB")
        elif self.current_filter == "Hue":
            hsv = img.convert("HSV")
            h, s, v = hsv.split()
            shift = int(intensity * 255 / 100)
            h = h.point(lambda i: (i + shift) % 256)
            hsv = Image.merge("HSV", (h, s, v))
            img = hsv.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        core_image = CoreImage(buffer, ext="png")
        self.ids.sample_image.texture = core_image.texture

    def update_total_augmented(self):
        try:
            image_files = [f for f in os.listdir(self.image_folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            num_images = len(image_files)
        except Exception:
            num_images = 0
        total_per_image = sum(int(v.get("count", 1)) for v in self.filter_settings.values())
        total_augmented = num_images * total_per_image
        self.ids.total_label.text = f"Total Augmented Images: {total_augmented}"

    def next_step(self):
        self.manager.current = 'augmentation'

# ------------------ Augmentation Screen ------------------
class AugmentationScreen(Screen):
    progress = NumericProperty(0)

    def on_pre_enter(self):
        self.update_total_images()

    def update_split_label(self, train_value):
        valid_value = 100 - int(train_value)
        self.ids.split_label.text = f"Train: {int(train_value)}% | Valid: {valid_value}%"
        self.update_total_images()

    def update_total_images(self):
        sample_screen = self.manager.get_screen('sample')
        try:
            image_files = [f for f in os.listdir(sample_screen.image_folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            num_images = len(image_files)
        except Exception:
            num_images = 0
        augmented_per_image = sum(int(v.get("count", 1)) for v in sample_screen.filter_settings.values())
        total_augmented = num_images * augmented_per_image
        if self.ids.copy_checkbox.active:
            total = total_augmented + num_images
        else:
            total = total_augmented
        train_percent = int(self.ids.train_slider.value)
        train_count = int(total * train_percent / 100)
        valid_count = total - train_count
        self.ids.total_augmented_label.text = f"Train Images: {train_count} | Valid Images: {valid_count}"

    def start_augmentation(self):
        threading.Thread(target=self.run_augmentation, daemon=True).start()

    def run_augmentation(self):
        train_percent = int(self.ids.train_slider.value)
        sample_screen = self.manager.get_screen('sample')
        settings = sample_screen.filter_settings
        image_folder = sample_screen.image_folder
        label_source_folder = self.manager.get_screen('folder_selection').label_folder
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_folder = f"augmenteddata-{date_str}"
        os.makedirs(output_folder, exist_ok=True)
        train_img = os.path.join(output_folder, "train", "images")
        train_lbl = os.path.join(output_folder, "train", "labels")
        valid_img = os.path.join(output_folder, "valid", "images")
        valid_lbl = os.path.join(output_folder, "valid", "labels")
        os.makedirs(train_img, exist_ok=True)
        os.makedirs(train_lbl, exist_ok=True)
        os.makedirs(valid_img, exist_ok=True)
        os.makedirs(valid_lbl, exist_ok=True)
        image_files = [f for f in os.listdir(image_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        total_steps = len(image_files) * sum(int(v.get("count", 1)) for v in settings.values())
        self.progress = 0
        all_augmented = []

        # Process only images with an existing label file.
        for idx, img_file in enumerate(image_files):
            original_label_path = os.path.join(label_source_folder, os.path.splitext(img_file)[0] + ".txt")
            if not os.path.exists(original_label_path):
                continue
            def augment_image(img_path):
                try:
                    img = Image.open(os.path.join(image_folder, img_path)).convert("RGB")
                except Exception:
                    return []
                augmented_files = []
                # Use a distinct variable 'j' for each augmentation count
                count = int(settings.get("Shake Blur", {}).get("count", 1)) if settings.get("Shake Blur") else 1
                # For each filter, use its own count value
                count = int(settings.get(filt, {}).get("count", 1)) if False else 1  # fallback below
                # We'll iterate using each filter's defined count
                for filt, params in settings.items():
                    count = int(params.get("count", 1))
                    intensity = int(params.get("intensity", 50))
                    for j in range(count):
                        aug = img.copy()
                        if filt == "Salt and Pepper":
                            np_img = np.array(aug)
                            row, col, ch = np_img.shape
                            s_vs_p = 0.5
                            amount = intensity / 500.0
                            num_salt = np.ceil(amount * np_img.size * s_vs_p)
                            num_pepper = np.ceil(amount * np_img.size * (1.0 - s_vs_p))
                            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in np_img.shape[:2]]
                            np_img[coords[0], coords[1], :] = 255
                            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in np_img.shape[:2]]
                            np_img[coords[0], coords[1], :] = 0
                            aug = Image.fromarray(np.uint8(np_img))
                        elif filt == "Brightness Increase":
                            enhancer = ImageEnhance.Brightness(aug)
                            factor = 1 + intensity/100.0
                            aug = enhancer.enhance(factor)
                        elif filt == "Brightness Decrease":
                            enhancer = ImageEnhance.Brightness(aug)
                            factor = max(0.1, 1 - intensity/100.0)
                            aug = enhancer.enhance(factor)
                        elif filt == "Saturation Increase":
                            enhancer = ImageEnhance.Color(aug)
                            factor = 1 + intensity/100.0
                            aug = enhancer.enhance(factor)
                        elif filt == "Blur":
                            aug = aug.filter(ImageFilter.GaussianBlur(radius=intensity/10.0))
                        elif filt == "Sunlight":
                            width, height = aug.size
                            mask = Image.new("L", aug.size, 0)
                            draw = ImageDraw.Draw(mask)
                            points = [(random.randint(0, width-1), random.randint(0, height-1)) for _ in range(3)]
                            draw.polygon(points, fill=255)
                            enhancer = ImageEnhance.Brightness(aug)
                            bright_img = enhancer.enhance(1 + intensity/100.0)
                            aug = Image.composite(bright_img, aug, mask)
                        elif filt == "Shake Blur":
                            iterations = 5
                            max_offset = max(1, intensity // 10)
                            accum = None
                            # Use 'k' for inner loop
                            for k in range(iterations):
                                offset_x = random.randint(-max_offset, max_offset)
                                offset_y = random.randint(-max_offset, max_offset)
                                shifted = ImageChops.offset(aug, offset_x, offset_y)
                                if accum is None:
                                    accum = np.array(shifted, dtype=np.float32)
                                else:
                                    accum += np.array(shifted, dtype=np.float32)
                            accum = accum / iterations
                            aug = Image.fromarray(np.uint8(accum))
                        elif filt == "Shadow":
                            img_rgba = aug.convert("RGBA")
                            overlay = Image.new("RGBA", img_rgba.size, (0,0,0,0))
                            draw = ImageDraw.Draw(overlay)
                            num_polygons = random.randint(3, 5)
                            for _ in range(num_polygons):
                                num_vertices = random.randint(3, 6)
                                points = [(random.randint(0, img_rgba.size[0]-1), random.randint(0, img_rgba.size[1]-1)) for _ in range(num_vertices)]
                                alpha = int(50 + intensity/2)
                                if alpha > 200:
                                    alpha = 200
                                draw.polygon(points, fill=(0, 0, 0, alpha))
                            aug = Image.alpha_composite(img_rgba, overlay).convert("RGB")
                        elif filt == "Hue":
                            hsv = aug.convert("HSV")
                            h, s, v = hsv.split()
                            shift = int(intensity * 255 / 100)
                            h = h.point(lambda i: (i + shift) % 256)
                            hsv = Image.merge("HSV", (h, s, v))
                            aug = hsv.convert("RGB")
                        base, ext = os.path.splitext(img_path)
                        new_name = f"{filt.replace(' ', '_')}_{j}_{img_path}"
                        save_path = os.path.join(output_folder, new_name)
                        aug.save(save_path)
                        augmented_files.append((new_name, img_path))
                return augmented_files

            aug_list = augment_image(img_file)
            all_augmented.extend(aug_list)
            self.progress += len(aug_list)

            @mainthread
            def update_label():
                # Calculate remaining time by dividing the estimated time by 12 and showing integer seconds only
                remaining = int((max(0, total_steps - self.progress) * 0.5) / 12)
                progress_percent = min((self.progress / total_steps) * 100, 100)
                self.ids.progress_bar.value = progress_percent
                self.ids.progress_label.text = f"Progress: {progress_percent:.0f}% - Remaining: {remaining}s"
            update_label()

        # Move augmented images to train/valid folders, ensuring that the label is processed too.
        for (aug_name, orig_name) in all_augmented:
            original_label_path = os.path.join(label_source_folder, os.path.splitext(orig_name)[0] + ".txt")
            if not os.path.exists(original_label_path):
                continue
            if random.random() < train_percent / 100.0:
                dest_img = train_img
                dest_lbl = train_lbl
            else:
                dest_img = valid_img
                dest_lbl = valid_lbl
            src_path = os.path.join(output_folder, aug_name)
            dest_path = os.path.join(dest_img, aug_name)
            if os.path.exists(src_path):
                try:
                    shutil.move(src_path, dest_path)
                except Exception as e:
                    print(f"Error moving file {aug_name}: {e}")
            else:
                print(f"File {src_path} not found, skipping.")
            label_filename = os.path.splitext(aug_name)[0] + ".txt"
            with open(original_label_path, "r") as f:
                label_content = f.read()
            with open(os.path.join(dest_lbl, label_filename), "w") as f:
                f.write(label_content)

        # Copy original images and labels if the "copy original" option is active.
        if self.ids.copy_checkbox.active:
            for img_file in image_files:
                original_label_path = os.path.join(label_source_folder, os.path.splitext(img_file)[0] + ".txt")
                if not os.path.exists(original_label_path):
                    continue
                if random.random() < train_percent / 100.0:
                    dest_img = train_img
                    dest_lbl = train_lbl
                else:
                    dest_img = valid_img
                    dest_lbl = valid_lbl
                shutil.copy(os.path.join(image_folder, img_file), os.path.join(dest_img, img_file))
                shutil.copy(original_label_path, os.path.join(dest_lbl, os.path.basename(original_label_path)))

        @mainthread
        def finish():
            self.ids.progress_label.text = "Augmentation complete!"
            self.ids.finish_button.opacity = 1
            self.ids.finish_button.disabled = False
            def open_and_exit(instance):
                try:
                    os.startfile(output_folder)
                except AttributeError:
                    import subprocess
                    subprocess.call(["xdg-open", output_folder])
                App.get_running_app().stop()
            self.ids.finish_button.bind(on_release=open_and_exit)
        finish()

# ------------------ Main App ------------------
class AugmentorApp(App):
    def build(self):
        self.title = "Image Augmentor"
        return Builder.load_string("""
#:kivy 2.0
<LoadingScreen>:
    BoxLayout:
        orientation: 'vertical'
        spacing: 5
        canvas.before:
            Color:
                rgba: 0.1,0.1,0.1,1
            Rectangle:
                pos: self.pos
                size: self.size
        Image:
            source: "Auriga.png"
            size_hint: None, None
            size: 200,250
            pos_hint: {"center_x": 0.5}
        Label:
            text: "Augmentor"
            font_size: '32sp'
            color: 1,1,1,1
            size_hint_y: None
            height: '50dp'
            pos_hint: {"center_x": 0.5}
        Label:
            text: root.loading_text
            font_size: '30sp'
            color: 1,1,1,1
            pos_hint: {"center_x": 0.5}
<FolderSelectionScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10
        canvas.before:
            Color:
                rgba: 0.1,0.1,0.1,1
            Rectangle:
                pos: self.pos
                size: self.size
        Label:
            text: "Select Image Folder"
            color: 1,1,1,1
        BoxLayout:
            size_hint_y: None
            height: '40dp'
            spacing: 10
            TextInput:
                id: image_folder_input
                text: root.image_folder
                multiline: False
                hint_text: "Enter path to images folder"
                background_color: 0.2,0.2,0.2,1
                foreground_color: 1,1,1,1
            Button:
                text: "Browse"
                size_hint_x: None
                width: '80dp'
                on_release: root.browse_folder('image')
        Label:
            id: image_count_label
            text: "Image files: " + str(root.image_count)
            color: 1,1,1,1
        Label:
            text: "Select Label Folder"
            color: 1,1,1,1
        BoxLayout:
            size_hint_y: None
            height: '40dp'
            spacing: 10
            TextInput:
                id: label_folder_input
                text: root.label_folder
                multiline: False
                hint_text: "Enter path to labels folder"
                background_color: 0.2,0.2,0.2,1
                foreground_color: 1,1,1,1
            Button:
                text: "Browse"
                size_hint_x: None
                width: '80dp'
                on_release: root.browse_folder('label')
        Label:
            id: label_count_label
            text: "Label files: " + str(root.label_count)
            color: 1,1,1,1
        Label:
            id: warning_label
            text: ""
            color: 1,0,0,1
        Button:
            text: "Apply"
            size_hint_y: None
            height: '50dp'
            on_release: root.validate_folders()
<SampleScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10
        canvas.before:
            Color:
                rgba: 0.1,0.1,0.1,1
            Rectangle:
                pos: self.pos
                size: self.size
        RelativeLayout:
            size_hint_y: 0.55
            Image:
                id: sample_image
                size_hint: None, None
                size: 800,800
                pos_hint: {"center_x":0.5, "center_y":0.5}
                allow_stretch: True
            Button:
                text: "<"
                size_hint: None, None
                size: 40,40
                pos_hint: {"x":0, "center_y":0.5}
                opacity: 0.7
                on_release: root.prev_image()
            Button:
                text: ">"
                size_hint: None, None
                size: 40,40
                pos_hint: {"right":1, "center_y":0.5}
                opacity: 0.7
                on_release: root.next_image()
        BoxLayout:
            orientation: 'vertical'
            size_hint_y: 0.45
            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '40dp'
                Spinner:
                    id: filter_spinner
                    text: "Salt and Pepper"
                    values: ["Salt and Pepper", "Brightness Increase", "Brightness Decrease", "Saturation Increase", "Blur", "Sunlight", "Shake Blur", "Shadow", "Hue"]
                    on_text: root.on_filter_change(self.text)
                Button:
                    text: "Next Filter"
                    size_hint_x: None
                    width: '100dp'
                    on_release: root.next_filter()
            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '40dp'
                Label:
                    text: "Count:"
                    size_hint_x: None
                    width: '40dp'
                    color: 1,1,1,1
                TextInput:
                    id: filter_count
                    text: "1"
                    multiline: False
                    input_filter: 'int'
                    size_hint_x: None
                    width: '40dp'
                    background_color: 0.2,0.2,0.2,1
                    foreground_color: 1,1,1,1
                    on_text: root.save_current_filter_settings()
                Label:
                    text: "Intensity:"
                    size_hint_x: None
                    width: '70dp'
                    color: 1,1,1,1
                Slider:
                    id: filter_slider
                    min: 0
                    max: 100
                    value: 50
                    on_value: root.on_slider_change(self.value)
            Label:
                id: total_label
                text: "Total Augmented Images: 0"
                size_hint_y: None
                height: '30dp'
                color: 1,1,1,1
            Button:
                text: "Next Step"
                size_hint_y: None
                height: '50dp'
                on_release: root.next_step()
<AugmentationScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10
        canvas.before:
            Color:
                rgba: 0.1,0.1,0.1,1
            Rectangle:
                pos: self.pos
                size: self.size
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: None
            height: '40dp'
            Label:
                text: "Train (%):"
                size_hint_x: None
                width: '80dp'
                color: 1,1,1,1
            Slider:
                id: train_slider
                min: 0
                max: 100
                value: 80
                on_value: root.update_split_label(self.value)
            Label:
                id: split_label
                text: "Train: 80% | Valid: 20%"
                size_hint_x: None
                width: '200dp'
                color: 1,1,1,1
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: None
            height: '40dp'
            spacing: 5
            CheckBox:
                id: copy_checkbox
                active: False
                on_active: root.update_total_images()
            Label:
                text: "Copy original data?"
                color: 1,1,1,1
        Label:
            id: total_augmented_label
            text: "Total Images: 0"
            size_hint_y: None
            height: '30dp'
            color: 1,1,1,1
        Button:
            text: "Start Augmentation"
            size_hint_y: None
            height: '50dp'
            on_release: root.start_augmentation()
        Button:
            id: finish_button
            text: "Finish"
            size_hint_y: None
            height: '50dp'
            background_color: 0,1,0,1
            opacity: 0
            disabled: True
        ProgressBar:
            id: progress_bar
            max: 100
            value: 0
        Label:
            id: progress_label
            text: "Progress: 0% - Remaining: 0s"
            size_hint_y: None
            height: '30dp'
            color: 1,1,1,1
ScreenManager:
    LoadingScreen:
        name: "loading"
    FolderSelectionScreen:
        name: "folder_selection"
    SampleScreen:
        name: "sample"
    AugmentationScreen:
        name: "augmentation"
""")
        
if __name__ == '__main__':
    AugmentorApp().run()
