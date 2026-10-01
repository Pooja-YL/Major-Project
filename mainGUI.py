import os
import shutil
import tempfile
import uuid
from datetime import datetime
from tkinter import filedialog

import customtkinter as ctk
import pyautogui
import pygetwindow
import tensorflow as tf
from PIL import Image

from fracture_type_model import predict_fracture_type
from medical_report import generate_medical_report
from predictions import enhance_xray_image, generate_fracture_localization, generate_gradcam, get_image_quality_report, predict


project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = project_folder + '/images/'
filename = ""

COLORS = {
    "navy": "#123B66", "navy_light": "#1B4F7A", "blue": "#1976D2",
    "cyan": "#00A6A6", "cyan_light": "#E1F5F5", "page": "#F4F8FC",
    "white": "#FFFFFF", "ink": "#172B4D", "muted": "#61758D",
    "line": "#D9E5EF", "green": "#16A085", "green_light": "#E4F5F0",
    "red": "#E74C3C", "red_light": "#FCEAE7", "warning": "#F4A62A",
    "warning_ink": "#946000",
    "warning_light": "#FFF4DF", "image_bg": "#EAF2F8",
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.title("Bone Fracture Detection | BoneSight Dashboard")
        self.geometry("1400x900")
        self.minsize(1120, 760)
        self.configure(fg_color=COLORS["page"])
        self.predicted_bone_type = None
        self.prediction_history = []
        self.current_result = None
        self.fracture_type_prediction = None
        self.fracture_type_confidence = None
        self.predicted_bone_confidence = None
        self.fracture_confidence = None
        self.prediction_id = None
        self.report_id = None
        self.report_generated_at = None
        self.medical_report_status_text = "No report has been generated yet."
        self.image_quality_report = None
        self.gradcam_image = None
        self.generated_pdf_path = None
        self.localized_image = None
        self.analysis_filename = ""
        self.enhanced_filename = ""
        self.pages = {}
        self.nav_buttons = {}
        self.preview_scroll_enabled = False
        self.comparison_scroll_enabled = False
        self._build_sidebar()
        self._build_dashboard()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=COLORS["navy"])
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.grid_columnconfigure(0, weight=0)
        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=22, pady=(26, 34))
        mark = ctk.CTkFrame(brand, width=38, height=38, fg_color=COLORS["cyan"], corner_radius=11)
        mark.pack(side="left", padx=(0, 11))
        mark.pack_propagate(False)
        ctk.CTkLabel(mark, text="+", text_color=COLORS["white"], font=ctk.CTkFont(size=26, weight="bold")).pack(expand=True)
        brand_text = ctk.CTkFrame(brand, fg_color="transparent")
        brand_text.pack(anchor="w")
        ctk.CTkLabel(brand_text, text="BONESIGHT", text_color=COLORS["white"], font=ctk.CTkFont(size=17, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(brand_text, text="AI RADIOLOGY WORKSPACE", text_color="#A9C2D5", font=ctk.CTkFont(size=8, weight="bold")).pack(anchor="w", pady=(2, 0))
        ctk.CTkLabel(self.sidebar, text="WORKSPACE", text_color="#9AB2C7", font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w", padx=24, pady=(0, 10))
        self._nav_button("Overview", "⌂", True, self.show_overview)
        self._nav_button("Prediction History", "◷", False, self.show_prediction_history)
        self._nav_button("Clinical Guide", "▤", False, self.show_clinical_guide)
        self._nav_button("Image Comparison", "⇄", False, self.show_image_comparison)
        self._nav_button("Grad-CAM", "◉", False, self.show_gradcam)
        self._nav_button("Fracture Localization", "⌖", False, self.show_localization_page)
        self._nav_button(
            "Fracture Type Classification Model",
            "◇",
            False,
            self.show_fracture_type_page,
            multiline=True,
        )
        self._nav_button("PDF Report", "▧", False, self.show_medical_report_page)
        ctk.CTkFrame(self.sidebar, height=1, fg_color="#315573").pack(fill="x", padx=20, pady=(22, 14))
        ctk.CTkLabel(self.sidebar, text="CLINICAL DECISION SUPPORT", text_color="#A9C2D5", font=ctk.CTkFont(size=9, weight="bold")).pack(anchor="w", padx=22)
        ctk.CTkLabel(self.sidebar, text="AI findings support review and\ndo not replace clinical judgment.", text_color="#A9C2D5", justify="left", anchor="w", font=ctk.CTkFont(size=10)).pack(anchor="w", padx=22, pady=(6, 0))

    def _nav_button(self, text, icon, active=False, command=None, multiline=False):
        button_text = f"  {icon}      {text}"
        if multiline:
            button_text = f"  {icon}      Fracture Type\n           Classification Model"
        button = ctk.CTkButton(self.sidebar, text=button_text, anchor="w", height=56 if multiline else 46, corner_radius=9,
                               fg_color=COLORS["cyan"] if active else "transparent", hover_color=COLORS["navy_light"],
                               text_color=COLORS["white"], font=ctk.CTkFont(size=12, weight="bold"),
                               command=command or (lambda: self._set_status(f"{text} is available in this workspace")))
        button.pack(fill="x", padx=12, pady=3)
        self.nav_buttons[text] = button

    def _build_dashboard(self):
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        app_header = ctk.CTkFrame(self, fg_color=COLORS["white"], corner_radius=0, height=76, border_width=1, border_color=COLORS["line"])
        app_header.grid(row=0, column=1, sticky="ew")
        app_header.grid_propagate(False)
        app_header.grid_columnconfigure(0, weight=1)
        header_identity = ctk.CTkFrame(app_header, fg_color="transparent")
        header_identity.pack(side="left", padx=30, pady=16)
        ctk.CTkLabel(header_identity, text="BONESIGHT  /  RADIOLOGY WORKSPACE", text_color=COLORS["ink"], font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        self.header_study_label = ctk.CTkLabel(header_identity, text="No study selected", text_color=COLORS["muted"], font=ctk.CTkFont(size=10))
        self.header_study_label.pack(anchor="w", pady=(3, 0))
        status = ctk.CTkFrame(app_header, fg_color=COLORS["green_light"], corner_radius=8)
        status.pack(side="right", padx=30, pady=19)
        ctk.CTkLabel(status, text="●", text_color=COLORS["green"], font=ctk.CTkFont(size=12)).pack(side="left", padx=(11, 5), pady=7)
        ctk.CTkLabel(status, text="LOCAL AI WORKSPACE", text_color=COLORS["navy"], font=ctk.CTkFont(size=9, weight="bold")).pack(side="left", padx=(0, 11))
        self.content = ctk.CTkScrollableFrame(
            self,
            fg_color=COLORS["page"],
            corner_radius=0,
            scrollbar_fg_color=COLORS["page"],
            scrollbar_button_color=COLORS["cyan"],
            scrollbar_button_hover_color=COLORS["navy_light"],
        )
        self.content.grid(row=1, column=1, sticky="nsew", padx=30, pady=(18, 26))
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)
        self.overview_page = ctk.CTkFrame(self.content, fg_color="transparent")
        self.overview_page.grid(row=0, column=0, sticky="nsew")
        self.overview_page.grid_columnconfigure(0, weight=1)
        self.overview_page.grid_rowconfigure(2, weight=1)
        self.pages["overview"] = self.overview_page
        topbar = ctk.CTkFrame(self.overview_page, fg_color="transparent")
        topbar.grid(row=0, column=0, sticky="ew", pady=(0, 22))
        topbar.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(topbar, text="Diagnostic overview", text_color=COLORS["ink"], font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(topbar, text="Review X-ray evidence with transparent AI assistance", text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", pady=(4, 0))
        status = ctk.CTkFrame(topbar, fg_color=COLORS["white"], corner_radius=10)
        status.grid(row=0, column=1, rowspan=2, sticky="e")
        ctk.CTkLabel(status, text="●", text_color=COLORS["green"], font=ctk.CTkFont(size=15)).pack(side="left", padx=(14, 6), pady=10)
        ctk.CTkLabel(status, text="Inference engine online", text_color=COLORS["ink"], font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=(0, 14))
        self._build_stat_cards()
        self._build_workspace()
        self._build_fracture_type_section()
        self._build_fracture_type_page()
        self._build_medical_report_page()
        self._build_localization_page()
        self._build_gradcam_page()
        self._build_history()

    def _build_stat_cards(self):
        cards = ctk.CTkFrame(self.overview_page, fg_color="transparent")
        cards.grid(row=1, column=0, sticky="ew", pady=(0, 18))
        for column in range(2):
            cards.grid_columnconfigure(column, weight=1)
        self.images_stat = self._stat_card(cards, 0, "IMAGES REVIEWED", "0", "This session", COLORS["green"], "▣")
        self.confidence_stat = self._stat_card(cards, 1, "LATEST CONFIDENCE", "--", "Awaiting analysis", COLORS["warning"], "◒")

    def _stat_card(self, parent, column, eyebrow, value, detail, accent, icon):
        card = ctk.CTkFrame(parent, fg_color=COLORS["white"], corner_radius=12, border_width=1, border_color=COLORS["line"])
        card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 7, 7 if column == 0 else 0))
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text=icon, text_color=accent, font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=1, rowspan=2, padx=15, pady=14)
        ctk.CTkLabel(card, text=eyebrow, text_color=COLORS["muted"], font=ctk.CTkFont(size=9, weight="bold")).grid(row=0, column=0, sticky="w", padx=16, pady=(13, 0))
        value_label = ctk.CTkLabel(card, text=value, text_color=COLORS["ink"], font=ctk.CTkFont(size=20, weight="bold"))
        value_label.grid(row=1, column=0, sticky="w", padx=16)
        ctk.CTkLabel(card, text=detail, text_color=COLORS["muted"], font=ctk.CTkFont(size=10)).grid(row=2, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 13))
        return value_label

    def _build_workspace(self):
        workspace = ctk.CTkFrame(self.overview_page, fg_color="transparent")
        workspace.grid(row=2, column=0, sticky="nsew", pady=(0, 18))
        workspace.grid_columnconfigure(0, weight=6, uniform="workspace")
        workspace.grid_columnconfigure(1, weight=5, uniform="workspace")
        workspace.grid_rowconfigure(0, weight=1)
        upload_card = ctk.CTkFrame(workspace, fg_color=COLORS["white"], corner_radius=14, border_width=1, border_color=COLORS["line"])
        upload_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        upload_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(upload_card, text="X-RAY STUDY", text_color=COLORS["muted"], font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, sticky="w", padx=22, pady=(20, 2))
        ctk.CTkLabel(upload_card, text="Upload an image for analysis", text_color=COLORS["ink"], font=ctk.CTkFont(size=18, weight="bold")).grid(row=1, column=0, sticky="w", padx=22)
        self.frame2 = ctk.CTkScrollableFrame(upload_card, fg_color=COLORS["image_bg"], corner_radius=10, height=360)
        self.frame2.grid(row=2, column=0, sticky="nsew", padx=22, pady=16)
        self.frame2.grid_columnconfigure(0, weight=1)
        self.frame2.grid_rowconfigure(0, weight=1)
        self.preview_label = ctk.CTkLabel(self.frame2, text="", image=None)
        self.preview_label.pack(expand=True, padx=12, pady=12)
        self.original_image_title = ctk.CTkLabel(self.frame2, text="Original Image", text_color=COLORS["ink"], font=ctk.CTkFont(size=11, weight="bold"))
        self.original_image_title.pack(anchor="center", pady=(8, 0))
        self.set_placeholder_image()
        actions = ctk.CTkFrame(upload_card, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", padx=22, pady=(0, 20))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        self.upload_btn = ctk.CTkButton(actions, text="+  Upload X-ray", height=38, corner_radius=8, fg_color=COLORS["navy"], hover_color=COLORS["navy_light"], font=ctk.CTkFont(size=12, weight="bold"), command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.predict_btn = ctk.CTkButton(actions, text="Run Analysis", height=40, corner_radius=8, fg_color=COLORS["cyan"], hover_color="#078D8D", text_color=COLORS["white"], font=ctk.CTkFont(size=12, weight="bold"), command=self.predict_gui)
        self.predict_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.preview_toggle_btn = ctk.CTkButton(upload_card, text="Scroll view", height=28, corner_radius=8, fg_color="transparent", border_width=1, border_color=COLORS["line"], text_color=COLORS["ink"], hover_color=COLORS["cyan_light"], font=ctk.CTkFont(size=10, weight="bold"), command=self._toggle_overview_scroll)
        self.preview_toggle_btn.grid(row=4, column=0, sticky="ew", padx=22, pady=(0, 18))

        result_card = ctk.CTkFrame(workspace, fg_color=COLORS["white"], corner_radius=14, border_width=1, border_color=COLORS["line"])
        result_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        result_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(result_card, text="AI ASSESSMENT", text_color=COLORS["muted"], font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, sticky="w", padx=22, pady=(20, 2))
        ctk.CTkLabel(result_card, text="Prediction results", text_color=COLORS["ink"], font=ctk.CTkFont(size=18, weight="bold")).grid(row=1, column=0, sticky="w", padx=22)
        self.result_badge = ctk.CTkLabel(result_card, text="Awaiting image", text_color=COLORS["muted"], fg_color=COLORS["image_bg"], corner_radius=8, width=150, height=34, font=ctk.CTkFont(size=13, weight="bold"))
        self.result_badge.grid(row=2, column=0, sticky="w", padx=22, pady=(18, 20))
        self.res1_label = ctk.CTkLabel(result_card, text="Body part  --", anchor="w", text_color=COLORS["ink"], font=ctk.CTkFont(size=14, weight="bold"))
        self.res1_label.grid(row=3, column=0, sticky="ew", padx=22)
        self.res1_conf_label = ctk.CTkLabel(result_card, text="Part confidence  --", anchor="w", text_color=COLORS["muted"], font=ctk.CTkFont(size=11))
        self.res1_conf_label.grid(row=4, column=0, sticky="ew", padx=22, pady=(2, 4))
        self.part_progress = ctk.CTkProgressBar(result_card, height=8, corner_radius=4, progress_color=COLORS["cyan"], fg_color=COLORS["cyan_light"])
        self.part_progress.grid(row=5, column=0, sticky="ew", padx=22, pady=(0, 16))
        self.res2_label = ctk.CTkLabel(result_card, text="Fracture status  --", anchor="w", text_color=COLORS["ink"], font=ctk.CTkFont(size=14, weight="bold"))
        self.res2_label.grid(row=6, column=0, sticky="ew", padx=22)
        self.res2_conf_label = ctk.CTkLabel(result_card, text="Fracture confidence  --", anchor="w", text_color=COLORS["muted"], font=ctk.CTkFont(size=11))
        self.res2_conf_label.grid(row=7, column=0, sticky="ew", padx=22, pady=(2, 4))
        self.fracture_progress = ctk.CTkProgressBar(result_card, height=8, corner_radius=4, progress_color=COLORS["green"], fg_color=COLORS["green_light"])
        self.fracture_progress.grid(row=8, column=0, sticky="ew", padx=22, pady=(0, 18))
        self.gradcam_btn = ctk.CTkButton(result_card, text="◎  Show Grad-CAM", height=36, corner_radius=8, fg_color=COLORS["navy"], hover_color=COLORS["navy_light"], state="disabled", font=ctk.CTkFont(size=11, weight="bold"), command=self.show_gradcam)
        self.gradcam_btn.grid(row=9, column=0, sticky="ew", padx=22, pady=(0, 7))
        self.localization_btn = ctk.CTkButton(result_card, text="Show Fracture Localization", height=34, corner_radius=8, fg_color=COLORS["red_light"], hover_color="#F8D7D4", text_color=COLORS["red"], state="disabled", font=ctk.CTkFont(size=11, weight="bold"), command=self.show_fracture_localization)
        self.localization_btn.grid(row=10, column=0, sticky="ew", padx=22, pady=(0, 7))
        self.save_btn = ctk.CTkButton(result_card, text="Save Result", height=32, corner_radius=8, fg_color="transparent", border_width=1, border_color=COLORS["line"], text_color=COLORS["ink"], hover_color=COLORS["page"], state="disabled", font=ctk.CTkFont(size=11, weight="bold"), command=self.save_result)
        self.save_btn.grid(row=11, column=0, sticky="ew", padx=22, pady=(0, 6))
        self.save_label = ctk.CTkLabel(result_card, text="", text_color=COLORS["green"], font=ctk.CTkFont(size=10, weight="bold"))
        self.quality_label = ctk.CTkLabel(result_card, text="X-RAY IMAGE QUALITY\nNot checked", justify="left", anchor="w", text_color=COLORS["muted"], font=ctk.CTkFont(size=10, weight="bold"))
        self.quality_label.grid(row=12, column=0, sticky="ew", padx=22, pady=(0, 8))
        self.save_label.grid(row=13, column=0, sticky="ew", padx=22, pady=(0, 10))
        self.generate_pdf_btn = ctk.CTkButton(result_card, text="Generate PDF Report", height=34, corner_radius=8, fg_color=COLORS["cyan"], hover_color="#078D8D", text_color=COLORS["white"], state="disabled", font=ctk.CTkFont(size=11, weight="bold"), command=self.generate_pdf_report)
        self.generate_pdf_btn.grid(row=14, column=0, sticky="ew", padx=22, pady=(0, 6))
        self.save_pdf_btn = ctk.CTkButton(result_card, text="Save PDF Report", height=32, corner_radius=8, fg_color="transparent", border_width=1, border_color=COLORS["line"], text_color=COLORS["ink"], hover_color=COLORS["page"], state="disabled", font=ctk.CTkFont(size=11, weight="bold"), command=self.save_pdf_report)
        self.save_pdf_btn.grid(row=15, column=0, sticky="ew", padx=22, pady=(0, 10))
        self.part_progress.set(0)
        self.fracture_progress.set(0)

    def _build_history(self):
        history_card = ctk.CTkFrame(self.overview_page, fg_color=COLORS["white"], corner_radius=12, border_width=1, border_color=COLORS["line"])
        history_card.grid(row=4, column=0, sticky="ew")
        history_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(history_card, text="RECENT ANALYSIS", text_color=COLORS["muted"], font=ctk.CTkFont(size=9, weight="bold")).grid(row=0, column=0, sticky="w", padx=18, pady=(12, 0))
        self.history_label = ctk.CTkLabel(history_card, text="No studies analyzed in this session yet.", text_color=COLORS["muted"], anchor="w", font=ctk.CTkFont(size=11))
        self.history_label.grid(row=1, column=0, sticky="ew", padx=18, pady=(3, 12))

    def _build_fracture_type_section(self):
        card = ctk.CTkFrame(self.overview_page, fg_color=COLORS["white"], corner_radius=14, border_width=1, border_color=COLORS["line"])
        card.grid(row=3, column=0, sticky="ew", pady=(0, 18))
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text="FRACTURE TYPE CLASSIFICATION", text_color=COLORS["muted"], font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, sticky="w", padx=22, pady=(18, 2))
        ctk.CTkLabel(card, text="Fracture type", text_color=COLORS["ink"], font=ctk.CTkFont(size=18, weight="bold")).grid(row=1, column=0, sticky="w", padx=22)
        self.fracture_type_status = ctk.CTkLabel(
            card,
            text="Run analysis to see the experimental six-class prediction.",
            text_color=COLORS["red"],
            anchor="w",
            wraplength=850,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.fracture_type_status.grid(row=2, column=0, sticky="ew", padx=22, pady=(12, 6))
        ctk.CTkLabel(
            card,
            text="Classes: Transverse  |  Oblique  |  Spiral  |  Comminuted  |  Greenstick  |  Hairline",
            text_color=COLORS["muted"],
            anchor="w",
            wraplength=850,
            font=ctk.CTkFont(size=11),
        ).grid(row=3, column=0, sticky="ew", padx=22, pady=(0, 16))

    def _update_fracture_type_status(self):
        if self.current_result == "fractured":
            try:
                self.fracture_type_prediction, self.fracture_type_confidence = predict_fracture_type(
                    self.analysis_filename
                )
            except (OSError, ValueError, RuntimeError, tf.errors.OpError) as error:
                self.fracture_type_prediction = None
                self.fracture_type_confidence = None
                status = f"Fracture detected. Type classification unavailable: {error}"
                page_status = "The fracture type result is temporarily unavailable."
                predicted_type = "Unavailable"
                confidence = "--"
            else:
                status = (
                    f"Predicted type: {self.fracture_type_prediction}  |  "
                    f"Confidence: {self.fracture_type_confidence:.1f}%"
                )
                page_status = "Six-class analysis complete for this fracture-positive image."
                predicted_type = self.fracture_type_prediction
                confidence = f"{self.fracture_type_confidence:.1f}%"
        elif self.current_result == "normal":
            self.fracture_type_prediction = None
            self.fracture_type_confidence = None
            status = "No fracture detected; fracture type is not applicable."
            page_status = "The fracture detector found no fracture; subtype analysis was not run."
            predicted_type = "Not applicable"
            confidence = "--"
        else:
            self.fracture_type_prediction = None
            self.fracture_type_confidence = None
            status = "Run analysis to check for a fracture. Type classification is unavailable with the current model."
            page_status = "Upload an X-ray and run analysis to see the predicted type."
            predicted_type = "Awaiting analysis"
            confidence = "--"
        self.fracture_type_status.configure(text=status)
        self.fracture_type_page_status.configure(text=page_status)
        self.fracture_type_value.configure(text=predicted_type)
        self.fracture_type_confidence_value.configure(text=confidence)
        confidence_fraction = (
            max(0.0, min(1.0, self.fracture_type_confidence / 100.0))
            if self.current_result == "fractured" and self.fracture_type_confidence is not None
            else 0.0
        )
        self.fracture_type_confidence_bar.set(confidence_fraction)
        self.fracture_type_summary.configure(
            text=(
                f"Most likely pattern: {predicted_type}. The model score is {confidence}."
                if self.fracture_type_prediction
                else page_status
            )
        )
        for class_name, card in self.fracture_type_cards.items():
            selected = class_name == self.fracture_type_prediction
            card.configure(
                fg_color=COLORS["cyan_light"] if selected else COLORS["white"],
                border_width=2 if selected else 1,
                border_color=COLORS["cyan"] if selected else COLORS["line"],
            )

    def _build_fracture_type_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["fracture_type"] = page
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="Fracture Type Classification Model", text_color=COLORS["ink"], font=ctk.CTkFont(size=27, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, text="AI-assisted fracture pattern review", text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ctk.CTkLabel(header, text="06  /  ANALYSIS", text_color=COLORS["navy"], fg_color=COLORS["cyan_light"], corner_radius=7, font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0), ipadx=10, ipady=7)

        hero = ctk.CTkFrame(page, fg_color=COLORS["navy"], corner_radius=14)
        hero.grid(row=1, column=0, sticky="ew", pady=(0, 18))
        hero.grid_columnconfigure(0, weight=3)
        hero.grid_columnconfigure(1, weight=2)
        ctk.CTkLabel(hero, text="AI ANALYSIS  /  PREDICTED FRACTURE TYPE", text_color="#8FE3E3", font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, sticky="w", padx=24, pady=(22, 5))
        self.fracture_type_value = ctk.CTkLabel(hero, text="Awaiting analysis", text_color=COLORS["white"], anchor="w", font=ctk.CTkFont(size=32, weight="bold"))
        self.fracture_type_value.grid(row=1, column=0, sticky="w", padx=24)
        self.fracture_type_page_status = ctk.CTkLabel(hero, text="Upload an X-ray and run analysis to see the predicted type.", text_color="#D1E1EC", anchor="w", font=ctk.CTkFont(size=11))
        self.fracture_type_page_status.grid(row=2, column=0, sticky="w", padx=24, pady=(5, 22))
        confidence_panel = ctk.CTkFrame(hero, fg_color=COLORS["navy_light"], corner_radius=10)
        confidence_panel.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=(8, 20), pady=20)
        confidence_panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(confidence_panel, text="MODEL CONFIDENCE", text_color="#D1E1EC", font=ctk.CTkFont(size=9, weight="bold")).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 2))
        self.fracture_type_confidence_value = ctk.CTkLabel(confidence_panel, text="--", text_color=COLORS["white"], anchor="w", font=ctk.CTkFont(size=26, weight="bold"))
        self.fracture_type_confidence_value.grid(row=1, column=0, sticky="w", padx=18)
        self.fracture_type_confidence_bar = ctk.CTkProgressBar(confidence_panel, height=10, corner_radius=5, progress_color=COLORS["cyan"], fg_color="#54718B")
        self.fracture_type_confidence_bar.grid(row=2, column=0, sticky="ew", padx=18, pady=(10, 7))
        self.fracture_type_confidence_bar.set(0)
        ctk.CTkLabel(confidence_panel, text="Confidence score", text_color="#D1E1EC", font=ctk.CTkFont(size=10)).grid(row=3, column=0, sticky="w", padx=18, pady=(0, 15))

        ctk.CTkLabel(page, text="Fracture type library", text_color=COLORS["ink"], font=ctk.CTkFont(size=17, weight="bold")).grid(row=2, column=0, sticky="w", pady=(0, 10))
        type_grid = ctk.CTkFrame(page, fg_color="transparent")
        type_grid.grid(row=3, column=0, sticky="ew", pady=(0, 18))
        type_grid.grid_columnconfigure((0, 1, 2), weight=1, uniform="fracture-types")
        self.fracture_type_cards = {}
        type_details = (
            ("Transverse", "A break running across the bone."),
            ("Oblique", "A diagonal break through the bone."),
            ("Spiral", "A break that curves around the bone."),
            ("Comminuted", "A break that forms several fragments."),
            ("Greenstick", "An incomplete break, often bending one side."),
            ("Hairline", "A fine, narrow crack in the bone."),
        )
        for index, (class_name, description) in enumerate(type_details):
            row, column = divmod(index, 3)
            type_card = ctk.CTkFrame(type_grid, fg_color=COLORS["white"], corner_radius=10, border_width=1, border_color=COLORS["line"], height=112)
            type_card.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 6 if column < 2 else 0), pady=(0, 10))
            type_card.grid_propagate(False)
            ctk.CTkLabel(type_card, text=f"{index + 1:02d}", text_color=COLORS["navy"], fg_color=COLORS["cyan_light"], corner_radius=6, width=34, height=26, font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, sticky="w", padx=13, pady=(12, 4))
            ctk.CTkLabel(type_card, text=class_name, text_color=COLORS["ink"], font=ctk.CTkFont(size=13, weight="bold")).grid(row=1, column=0, sticky="w", padx=13)
            ctk.CTkLabel(type_card, text=description, text_color=COLORS["muted"], anchor="w", wraplength=245, font=ctk.CTkFont(size=10)).grid(row=2, column=0, sticky="ew", padx=13, pady=(3, 10))
            self.fracture_type_cards[class_name] = type_card

        detail_row = ctk.CTkFrame(page, fg_color="transparent")
        detail_row.grid(row=4, column=0, sticky="ew", pady=(0, 16))
        detail_row.grid_columnconfigure(0, weight=1, uniform="type-summary")
        detail_row.grid_columnconfigure(1, weight=1, uniform="type-summary")
        analysis_card = ctk.CTkFrame(detail_row, fg_color=COLORS["white"], corner_radius=10, border_width=1, border_color=COLORS["line"])
        analysis_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ctk.CTkLabel(analysis_card, text="AI ANALYSIS", text_color=COLORS["cyan"], font=ctk.CTkFont(size=9, weight="bold")).pack(anchor="w", padx=16, pady=(15, 5))
        ctk.CTkLabel(analysis_card, text="The model compares the fracture-positive X-ray with six supported fracture patterns and reports its leading class score.", text_color=COLORS["ink"], anchor="w", justify="left", wraplength=390, font=ctk.CTkFont(size=11)).pack(fill="x", padx=16, pady=(0, 15))
        summary_card = ctk.CTkFrame(detail_row, fg_color=COLORS["white"], corner_radius=10, border_width=1, border_color=COLORS["line"])
        summary_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ctk.CTkLabel(summary_card, text="CLASSIFICATION SUMMARY", text_color=COLORS["cyan"], font=ctk.CTkFont(size=9, weight="bold")).pack(anchor="w", padx=16, pady=(15, 5))
        self.fracture_type_summary = ctk.CTkLabel(summary_card, text="Upload an X-ray and run analysis to create a summary.", text_color=COLORS["ink"], anchor="w", justify="left", wraplength=390, font=ctk.CTkFont(size=11, weight="bold"))
        self.fracture_type_summary.pack(fill="x", padx=16, pady=(0, 15))

        ctk.CTkButton(page, text="↻  Analyze Another X-ray", height=40, corner_radius=8, fg_color=COLORS["cyan"], hover_color="#078D8D", text_color=COLORS["white"], font=ctk.CTkFont(size=12, weight="bold"), command=self._analyze_another_xray).grid(row=5, column=0, sticky="e", pady=(0, 8))

    def _build_medical_report_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["medical_report"] = page

        header = ctk.CTkFrame(page, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="PDF Report", text_color=COLORS["ink"], font=ctk.CTkFont(size=27, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, text="Generate and save a PDF summary for the current X-ray study.", text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ctk.CTkLabel(header, text="07  /  REPORTS", text_color=COLORS["navy"], fg_color=COLORS["cyan_light"], corner_radius=7, font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0), ipadx=10, ipady=7)

        report_card = ctk.CTkFrame(page, fg_color=COLORS["white"], corner_radius=14, border_width=1, border_color=COLORS["line"])
        report_card.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        report_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(report_card, text="CURRENT STUDY", text_color=COLORS["muted"], font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, sticky="w", padx=22, pady=(20, 3))
        self.medical_report_study_value = ctk.CTkLabel(report_card, text="No study selected", text_color=COLORS["ink"], anchor="w", font=ctk.CTkFont(size=18, weight="bold"))
        self.medical_report_study_value.grid(row=1, column=0, sticky="ew", padx=22)
        self.medical_report_result_value = ctk.CTkLabel(report_card, text="Run an analysis to prepare the study report.", text_color=COLORS["muted"], anchor="w", wraplength=840, font=ctk.CTkFont(size=12))
        self.medical_report_result_value.grid(row=2, column=0, sticky="ew", padx=22, pady=(5, 16))

        identifiers = ctk.CTkFrame(report_card, fg_color=COLORS["page"], corner_radius=8)
        identifiers.grid(row=3, column=0, sticky="ew", padx=22, pady=(0, 14))
        identifiers.grid_columnconfigure(0, weight=1)
        identifiers.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(identifiers, text="PREDICTION ID", text_color=COLORS["muted"], font=ctk.CTkFont(size=9, weight="bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 2))
        ctk.CTkLabel(identifiers, text="REPORT ID", text_color=COLORS["muted"], font=ctk.CTkFont(size=9, weight="bold")).grid(row=0, column=1, sticky="w", padx=14, pady=(10, 2))
        self.medical_report_prediction_id = ctk.CTkLabel(identifiers, text="Not assigned", text_color=COLORS["ink"], anchor="w", font=ctk.CTkFont(size=11, weight="bold"))
        self.medical_report_prediction_id.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        self.medical_report_report_id = ctk.CTkLabel(identifiers, text="Generated with PDF", text_color=COLORS["ink"], anchor="w", font=ctk.CTkFont(size=11, weight="bold"))
        self.medical_report_report_id.grid(row=1, column=1, sticky="ew", padx=14, pady=(0, 10))

        self.medical_report_status = ctk.CTkLabel(report_card, text=self.medical_report_status_text, text_color=COLORS["muted"], anchor="w", font=ctk.CTkFont(size=11, weight="bold"))
        self.medical_report_status.grid(row=4, column=0, sticky="ew", padx=22, pady=(0, 16))

        actions = ctk.CTkFrame(page, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        self.medical_report_generate_btn = ctk.CTkButton(actions, text="Generate PDF Report", height=42, corner_radius=8, fg_color=COLORS["cyan"], hover_color="#078D8D", text_color=COLORS["white"], state="disabled", font=ctk.CTkFont(size=12, weight="bold"), command=self.generate_pdf_report)
        self.medical_report_generate_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.medical_report_save_btn = ctk.CTkButton(actions, text="Save PDF Report", height=42, corner_radius=8, fg_color=COLORS["navy"], hover_color=COLORS["navy_light"], state="disabled", font=ctk.CTkFont(size=12, weight="bold"), command=self.save_pdf_report)
        self.medical_report_save_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self._update_medical_report_page()

    def _update_medical_report_page(self):
        if not hasattr(self, "medical_report_status"):
            return
        self.medical_report_study_value.configure(
            text=os.path.basename(filename) if filename else "No study selected"
        )
        if self.prediction_id and self.current_result:
            fracture_type = (
                self.fracture_type_prediction
                if self.current_result == "fractured" and self.fracture_type_prediction
                else "Not Applicable"
                if self.current_result == "normal"
                else "Unavailable"
            )
            self.medical_report_result_value.configure(
                text=(
                    f"{self.predicted_bone_type}  |  {self.current_result.capitalize()}  |  "
                    f"Fracture type: {fracture_type}"
                )
            )
        else:
            self.medical_report_result_value.configure(
                text="Run an analysis to prepare the study report."
            )
        self.medical_report_prediction_id.configure(
            text=self.prediction_id or "Not assigned"
        )
        self.medical_report_report_id.configure(
            text=self.report_id or "Generated with PDF"
        )
        self.medical_report_status.configure(text=self.medical_report_status_text)
        self.medical_report_generate_btn.configure(
            state="normal" if self.prediction_id else "disabled"
        )
        self.medical_report_save_btn.configure(
            state="normal"
            if self.generated_pdf_path and os.path.isfile(self.generated_pdf_path)
            else "disabled"
        )

    def _analyze_another_xray(self):
        self.show_overview()
        self.after(100, self.upload_image)

    def _build_localization_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)
        self.pages["localization"] = page
        ctk.CTkLabel(page, text="Fracture localization", text_color=COLORS["ink"], font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(page, text="Review the suspected fracture area highlighted on the uploaded X-ray.", text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", pady=(4, 22))
        card = ctk.CTkFrame(page, fg_color=COLORS["white"], corner_radius=14, border_width=1, border_color=COLORS["line"])
        card.grid(row=2, column=0, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text="Suspected fracture area", text_color=COLORS["ink"], font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, sticky="w", padx=22, pady=(20, 0))
        ctk.CTkLabel(card, text="The red box marks a high-activation region from the fracture model, not a confirmed fracture boundary.", text_color=COLORS["muted"], font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky="w", padx=22, pady=(3, 14))
        self.localization_status_label = ctk.CTkLabel(card, text="Upload an X-ray to view it here.", text_color=COLORS["muted"], anchor="w", font=ctk.CTkFont(size=11))
        self.localization_status_label.grid(row=2, column=0, sticky="ew", padx=22, pady=(0, 8))
        self.localization_image_label = ctk.CTkLabel(card, text="", image=None, fg_color=COLORS["image_bg"], corner_radius=10, height=470)
        self.localization_image_label.grid(row=3, column=0, sticky="nsew", padx=22, pady=(0, 20))

    def _show_localization_image(self, img):
        display = img.copy()
        display.thumbnail((900, 360), Image.Resampling.LANCZOS)
        self.localization_photo = ctk.CTkImage(light_image=display, dark_image=display, size=display.size)
        self.localization_image_label.configure(text="", image=self.localization_photo)

    def _build_gradcam_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["gradcam"] = page
        ctk.CTkLabel(page, text="Grad-CAM analysis", text_color=COLORS["ink"], font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(page, text="Review the image regions that influenced the fracture model output.", text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", pady=(4, 18))
        self.gradcam_summary = ctk.CTkFrame(page, fg_color=COLORS["cyan_light"], corner_radius=11)
        self.gradcam_summary.grid(row=2, column=0, sticky="ew", pady=(0, 18))
        self.gradcam_summary.grid_columnconfigure(0, weight=1)
        self.gradcam_summary_title = ctk.CTkLabel(self.gradcam_summary, text="Awaiting analysis", text_color=COLORS["navy"], anchor="w", font=ctk.CTkFont(size=14, weight="bold"))
        self.gradcam_summary_title.grid(row=0, column=0, sticky="ew", padx=18, pady=(13, 3))
        self.gradcam_summary_text = ctk.CTkLabel(self.gradcam_summary, text="Run an X-ray analysis to generate an explainability overlay.", text_color=COLORS["ink"], anchor="w", justify="left", wraplength=980, font=ctk.CTkFont(size=11))
        self.gradcam_summary_text.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 13))
        self.gradcam_images_frame = ctk.CTkFrame(page, fg_color="transparent")
        self.gradcam_images_frame.grid(row=3, column=0, sticky="ew")
        self.gradcam_images_frame.grid_columnconfigure((0, 1), weight=1, uniform="gradcam-images")
        self._refresh_gradcam_page()

    def _refresh_gradcam_page(self):
        if not hasattr(self, "gradcam_images_frame"):
            return
        for child in self.gradcam_images_frame.winfo_children():
            child.destroy()
        if self.gradcam_image is None or not self.analysis_filename or not os.path.isfile(self.analysis_filename) or not self.predicted_bone_type:
            self.gradcam_summary_title.configure(text="Awaiting analysis", text_color=COLORS["navy"])
            self.gradcam_summary_text.configure(text="Run an X-ray analysis to generate an explainability overlay.")
            ctk.CTkLabel(self.gradcam_images_frame, text="Grad-CAM images will appear here after analysis.", text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).grid(row=0, column=0, columnspan=2, sticky="ew", pady=60)
            return
        if self.current_result == "fractured":
            self.gradcam_summary_title.configure(text="Fracture pattern flagged", text_color=COLORS["red"])
            summary = f"The {self.predicted_bone_type.lower()} fracture model found image features associated with a fracture pattern. Highlighted regions influenced this result; use them to guide review, not as a confirmed fracture boundary."
        else:
            self.gradcam_summary_title.configure(text="No fracture pattern detected", text_color=COLORS["green"])
            summary = f"The {self.predicted_bone_type.lower()} fracture model did not identify a strong fracture pattern. The overlay shows regions that influenced the normal classification and does not replace professional review."
        self.gradcam_summary_text.configure(text=summary)
        with Image.open(self.analysis_filename) as original:
            original_display = original.convert("RGB")
            original_display.thumbnail((560, 560), Image.Resampling.LANCZOS)
            original_display = original_display.copy()
        overlay_display = self.gradcam_image.copy()
        overlay_display.thumbnail((560, 560), Image.Resampling.LANCZOS)
        self._add_explanation_image(self.gradcam_images_frame, "Original X-ray", original_display, 0)
        self._add_explanation_image(self.gradcam_images_frame, "Grad-CAM overlay", overlay_display, 1)

    def _update_fracture_localization(self):
        if not filename or not os.path.exists(filename):
            self.localization_status_label.configure(text="Upload an X-ray to view it here.")
            self.localization_image_label.configure(image=None)
            self.localization_photo = None
            return

        if self.current_result != "fractured":
            with Image.open(filename) as source_image:
                self._show_localization_image(source_image.convert("RGB"))
            status = (
                "No suspected fracture area to localize for this result."
                if self.current_result == "normal"
                else "X-ray loaded. Run analysis to locate a suspected area."
            )
            self.localization_status_label.configure(text=status)
            return

        if self.localized_image is not None:
            self.localization_status_label.configure(text="Red box shows the suspected high-activation region.")
            self._show_localization_image(self.localized_image)
            return

        try:
            self.localized_image = generate_fracture_localization(
                self.analysis_filename,
                self.predicted_bone_type,
                original_img=filename,
            )
        except Exception as error:
            self.localized_image = None
            with Image.open(filename) as source_image:
                self._show_localization_image(source_image.convert("RGB"))
            self.localization_status_label.configure(text=f"Localization unavailable: {error}")
            return
        self.localization_status_label.configure(text="Red box shows the suspected high-activation region.")
        self._show_localization_image(self.localized_image)

    def _show_page(self, page_name):
        if page_name == "history" and page_name not in self.pages:
            self._build_history_page()
        elif page_name == "guide" and page_name not in self.pages:
            self._build_clinical_guide()

        for page in self.pages.values():
            page.grid_forget()
        self.pages[page_name].grid(row=0, column=0, sticky="nsew")
        self._set_active_nav(page_name)
        if hasattr(self.content, "_parent_canvas"):
            self.content._parent_canvas.yview_moveto(0)

    def _set_active_nav(self, page_name):
        page_to_nav = {"overview": "Overview", "history": "Prediction History", "guide": "Clinical Guide", "comparison": "Image Comparison", "localization": "Fracture Localization", "gradcam": "Grad-CAM"}
        page_to_nav["fracture_type"] = "Fracture Type Classification Model"
        page_to_nav["medical_report"] = "PDF Report"
        for name, button in self.nav_buttons.items():
            button.configure(fg_color=COLORS["cyan"] if name == page_to_nav[page_name] else "transparent")

    def show_overview(self):
        self._show_page("overview")

    def show_prediction_history(self):
        self._show_page("history")
        self._refresh_history_page()

    def show_clinical_guide(self):
        self._show_page("guide")

    def show_image_comparison(self):
        if "comparison" not in self.pages:
            self._build_comparison_page()
        self._show_page("comparison")
        self._refresh_comparison_page()

    def show_fracture_type_page(self):
        self._show_page("fracture_type")

    def show_medical_report_page(self):
        self._show_page("medical_report")
        self._update_medical_report_page()

    def show_localization_page(self):
        self._show_page("localization")
        self._update_fracture_localization()

    def _build_comparison_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)
        self.pages["comparison"] = page
        ctk.CTkLabel(page, text="Image comparison", text_color=COLORS["ink"], font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(page, text="Compare the uploaded X-ray with the enhanced image used for analysis.", text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", pady=(4, 22))
        self.comparison_view = ctk.CTkScrollableFrame(page, fg_color=COLORS["white"], corner_radius=14, border_width=1, border_color=COLORS["line"])
        self.comparison_view.grid(row=2, column=0, sticky="nsew")
        self.comparison_view.grid_columnconfigure(0, weight=1)
        self.comparison_view.grid_columnconfigure(1, weight=1)
        self.comparison_toggle_btn = ctk.CTkButton(page, text="Scroll view", height=28, corner_radius=8, fg_color="transparent", border_width=1, border_color=COLORS["line"], text_color=COLORS["ink"], hover_color=COLORS["page"], font=ctk.CTkFont(size=10, weight="bold"), command=self._toggle_comparison_scroll)
        self.comparison_toggle_btn.grid(row=3, column=0, sticky="e", padx=(0, 16), pady=(8, 12))

    def _toggle_overview_scroll(self):
        self.preview_scroll_enabled = not self.preview_scroll_enabled
        self.preview_toggle_btn.configure(text="Fit view" if self.preview_scroll_enabled else "Scroll view")
        self._set_preview(Image.open(filename).convert("RGB")) if filename and os.path.exists(filename) else self.set_placeholder_image()

    def _toggle_comparison_scroll(self):
        self.comparison_scroll_enabled = not self.comparison_scroll_enabled
        self.comparison_toggle_btn.configure(text="Fit view" if self.comparison_scroll_enabled else "Scroll view")
        self._refresh_comparison_page()

    def _make_preview_container(self, parent, enabled):
        if enabled:
            container = ctk.CTkScrollableFrame(parent, fg_color="transparent", corner_radius=10)
            container.pack(fill="both", expand=True, padx=12, pady=12)
            return container
        container = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=10)
        container.pack(fill="both", expand=True, padx=12, pady=12)
        return container

    def _render_image_preview(self, parent, img, title):
        for child in parent.winfo_children():
            child.destroy()
        title_label = ctk.CTkLabel(parent, text=title, text_color=COLORS["ink"], font=ctk.CTkFont(size=11, weight="bold"))
        title_label.pack(anchor="center", pady=(8, 0))
        display = img.copy()
        max_width = 460 if not self.preview_scroll_enabled else 680
        max_height = 300 if not self.preview_scroll_enabled else 520
        display.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        photo = ctk.CTkImage(light_image=display, dark_image=display, size=display.size)
        image_label = ctk.CTkLabel(parent, text="", image=photo)
        image_label.pack(expand=True, padx=10, pady=(0, 10))
        setattr(image_label, "image", photo)

    def _render_comparison_preview(self, parent, original, enhanced):
        for child in parent.winfo_children():
            child.destroy()
        panel_grid = ctk.CTkFrame(parent, fg_color="transparent")
        panel_grid.pack(fill="both", expand=True, padx=8, pady=8)
        for column, (title, source_image) in enumerate((("Original image", original), ("Enhanced image", enhanced))):
            panel_grid.grid_columnconfigure(column, weight=1)
            panel = ctk.CTkFrame(panel_grid, fg_color="transparent")
            panel.grid(row=0, column=column, sticky="nsew", padx=12, pady=8)
            ctk.CTkLabel(panel, text=title, text_color=COLORS["ink"], font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(2, 10))
            display = source_image.copy()
            max_width = 360 if not self.comparison_scroll_enabled else 520
            max_height = 260 if not self.comparison_scroll_enabled else 420
            display.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            photo = ctk.CTkImage(light_image=display, dark_image=display, size=display.size)
            label = ctk.CTkLabel(panel, text="", image=photo)
            label.pack(expand=True)
            setattr(label, "image", photo)

    def _refresh_comparison_page(self):
        if not hasattr(self, "comparison_view"):
            return
        for child in self.comparison_view.winfo_children():
            child.destroy()
        if not filename or not self.enhanced_filename or not os.path.exists(self.enhanced_filename):
            ctk.CTkLabel(self.comparison_view, text="Upload an X-ray to compare the original and enhanced images.", text_color=COLORS["muted"], font=ctk.CTkFont(size=13)).pack(expand=True, padx=30, pady=60)
            return
        try:
            with Image.open(filename) as source_original, Image.open(self.enhanced_filename) as source_enhanced:
                comparison_container = self._make_preview_container(self.comparison_view, self.comparison_scroll_enabled)
                self._render_comparison_preview(comparison_container, source_original.convert("RGB"), source_enhanced.convert("RGB"))
        except (OSError, ValueError):
            ctk.CTkLabel(self.comparison_view, text="Image unavailable", text_color=COLORS["muted"]).pack(expand=True, pady=80)

    def _build_history_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)
        self.pages["history"] = page
        ctk.CTkLabel(page, text="Prediction history", text_color=COLORS["ink"], font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(page, text="Review the results generated during this application session.", text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", pady=(4, 22))
        self.history_rows = ctk.CTkFrame(page, fg_color=COLORS["white"], corner_radius=14, border_width=1, border_color=COLORS["line"])
        self.history_rows.grid(row=2, column=0, sticky="nsew")
        self.history_rows.grid_columnconfigure(0, weight=1)
        self.history_rows.grid_columnconfigure(1, weight=2)
        self.history_rows.grid_columnconfigure(2, weight=2)
        self.history_rows.grid_columnconfigure(3, weight=2)
        self.history_rows.grid_columnconfigure(4, weight=1)
        for column, heading in enumerate(("TIME", "BODY PART", "RESULT", "CONFIDENCE", "VIEW")):
            ctk.CTkLabel(self.history_rows, text=heading, text_color=COLORS["muted"], font=ctk.CTkFont(size=9, weight="bold"), anchor="w").grid(row=0, column=column, sticky="ew", padx=18, pady=(18, 10))
        self._refresh_history_page()

    def _refresh_history_page(self):
        if not hasattr(self, "history_rows"):
            return
        for child in self.history_rows.winfo_children():
            if int(child.grid_info().get("row", 0)) > 0:
                child.destroy()
        if not self.prediction_history:
            ctk.CTkLabel(self.history_rows, text="No predictions yet. Return to Overview to upload an X-ray.", text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).grid(row=1, column=0, columnspan=5, padx=18, pady=28)
            return
        for row, (time, part, result, confidence) in enumerate(reversed(self.prediction_history), start=1):
            text_color = COLORS["red"] if result == "fractured" else COLORS["green"]
            values = (time, part, result.capitalize(), f"{confidence:.2f}%")
            for column, value in enumerate(values):
                ctk.CTkLabel(self.history_rows, text=value, text_color=text_color if column == 2 else COLORS["ink"], font=ctk.CTkFont(size=11, weight="bold" if column == 2 else "normal"), anchor="w").grid(row=row, column=column, sticky="ew", padx=18, pady=12)
            ctk.CTkButton(self.history_rows, text="View", width=58, height=26, corner_radius=6, fg_color=COLORS["cyan_light"], hover_color="#C7ECEC", text_color=COLORS["navy"], font=ctk.CTkFont(size=10, weight="bold"), command=self.show_overview).grid(row=row, column=4, padx=18, pady=8)

    def _build_clinical_guide(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(3, weight=1)
        self.pages["guide"] = page
        ctk.CTkLabel(page, text="Clinical guide", text_color=COLORS["ink"], font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(page, text="How BoneSight processes and explains an X-ray prediction.", text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", pady=(4, 22))
        guide_grid = ctk.CTkFrame(page, fg_color="transparent")
        guide_grid.grid(row=2, column=0, sticky="ew")
        guide_grid.grid_columnconfigure(0, weight=1)
        guide_grid.grid_columnconfigure(1, weight=1)
        self._guide_card(guide_grid, 0, "01  Body-part classification", "The ResNet50 body-part model identifies whether the study most closely matches an elbow, hand, or shoulder X-ray. The part confidence indicates how strongly the model selected that category.")
        self._guide_card(guide_grid, 1, "02  Fracture classification", "The matching body-part model then compares the X-ray with fractured and normal patterns. The displayed confidence is the probability of the selected result.")
        self._guide_card(guide_grid, 2, "03  Grad-CAM review", "Grad-CAM highlights spatial regions that influenced the selected fracture result. It is an interpretation aid, not a pixel-level fracture boundary or a medical diagnosis.")
        self._guide_card(guide_grid, 3, "04  Review workflow", "Upload a clear X-ray, run the analysis, review both confidence values, inspect Grad-CAM regions, and use professional radiological judgment for the final decision.")

    def _guide_card(self, parent, index, title, description):
        row, column = divmod(index, 2)
        card = ctk.CTkFrame(parent, fg_color=COLORS["white"], corner_radius=12, border_width=1, border_color=COLORS["line"])
        card.grid(row=row, column=column, sticky="ew", padx=(0 if column == 0 else 7, 7 if column == 0 else 0), pady=(0, 14))
        ctk.CTkLabel(card, text=title, text_color=COLORS["ink"], font=ctk.CTkFont(size=13, weight="bold"), anchor="w").pack(fill="x", padx=18, pady=(16, 6))
        ctk.CTkLabel(card, text=description, text_color=COLORS["muted"], justify="left", anchor="w", wraplength=390, font=ctk.CTkFont(size=11)).pack(fill="x", padx=18, pady=(0, 17))

    def set_placeholder_image(self):
        self._set_preview(Image.open(folder_path + "Question_Mark.jpg").convert("RGB"))

    def _set_preview(self, img):
        for child in self.frame2.winfo_children():
            child.destroy()
        preview_container = self._make_preview_container(self.frame2, self.preview_scroll_enabled)
        self._render_image_preview(preview_container, img, "Original Image")

    def _set_comparison_preview(self, original, enhanced):
        for child in self.frame2.winfo_children():
            child.destroy()
        comparison = ctk.CTkFrame(self.frame2, fg_color="transparent")
        comparison.pack(fill="both", expand=True, padx=8, pady=8)
        for column, (title, source_image) in enumerate((("Poor Quality Image / Original", original), ("Good Quality Image / Enhanced", enhanced))):
            comparison.grid_columnconfigure(column, weight=1)
            panel = ctk.CTkFrame(comparison, fg_color="transparent")
            panel.grid(row=0, column=column, sticky="nsew", padx=4)
            ctk.CTkLabel(panel, text=title, text_color=COLORS["ink"], font=ctk.CTkFont(size=10, weight="bold")).pack(pady=(2, 5))
            display = source_image.copy()
            display.thumbnail((195, 215), Image.Resampling.LANCZOS)
            photo = ctk.CTkImage(light_image=display, dark_image=display, size=display.size)
            label = ctk.CTkLabel(panel, text="", image=photo)
            label.pack(expand=True)
            setattr(label, "image", photo)

    def _prepare_analysis_image(self):
        global filename
        try:
            quality_report = get_image_quality_report(filename)
        except (OSError, ValueError) as error:
            self._set_status(f"Image quality check failed: {error}")
            return False

        quality_issues = quality_report["issues"]
        quality_ok = not quality_issues
        quality_text = (
            f"X-RAY IMAGE QUALITY  |  {'PASS' if quality_ok else 'WARNING'}\n"
            f"Resolution {quality_report['width']}x{quality_report['height']}   "
            f"Brightness {quality_report['brightness']:.1f}   "
            f"Contrast {quality_report['contrast']:.1f}   "
            f"Sharpness {quality_report['sharpness']:.1f}"
        )
        self.analysis_filename = filename
        self.enhanced_filename = ""
        self.image_quality_report = quality_report
        self.quality_label.configure(
            text=quality_text,
            text_color=COLORS["green"] if quality_ok else COLORS["warning_ink"],
        )
        enhanced_path = os.path.join(tempfile.gettempdir(), "bonesight_enhanced_xray.png")
        try:
            enhance_xray_image(filename, enhanced_path)
            self.enhanced_filename = enhanced_path
            with Image.open(filename) as original_image:
                self._set_preview(original_image.convert("RGB"))
            if quality_ok:
                self.quality_label.configure(
                    text=quality_text + "\nGood Quality / PASS",
                    text_color=COLORS["green"],
                )
                self._set_status("Good Quality / PASS")
            else:
                self.analysis_filename = enhanced_path
                self.result_badge.configure(text="  POOR QUALITY DETECTED  ", text_color=COLORS["warning_ink"], fg_color=COLORS["warning_light"])
                self.quality_label.configure(
                    text=quality_text + "\nPoor Quality Detected\nGood Quality / PASS",
                    text_color=COLORS["warning_ink"],
                )
                self._set_status("Poor Quality Detected -> Good Quality / PASS")
            return True
        except (OSError, ValueError) as error:
            self._set_status(f"Image enhancement failed: {error}")
            return False

    def upload_image(self):
        global filename
        selected_filename = filedialog.askopenfilename(
            filetypes=[("X-ray images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All Files", "*.*")],
            initialdir=project_folder + '/test/'
        )
        if not selected_filename:
            return

        try:
            with Image.open(selected_filename) as img:
                img.verify()
        except (OSError, ValueError) as error:
            filename = ""
            self._set_status(f"Unable to open image: {error}")
            return

        filename = selected_filename
        self.header_study_label.configure(text=os.path.basename(filename))
        self.analysis_filename = filename
        self.save_label.configure(text="")
        self.quality_label.configure(text="X-RAY IMAGE QUALITY\nNot checked", text_color=COLORS["muted"])
        self.res2_label.configure(text="Fracture status  --")
        self.res2_conf_label.configure(text="Fracture confidence  --")
        self.res1_label.configure(text="Body part  --")
        self.res1_conf_label.configure(text="Part confidence  --")
        self.result_badge.configure(text="Awaiting analysis", text_color=COLORS["muted"], fg_color=COLORS["image_bg"])
        self.part_progress.set(0)
        self.fracture_progress.set(0)
        self.predicted_bone_type = None
        self.current_result = None
        self.predicted_bone_confidence = None
        self.fracture_confidence = None
        self.prediction_id = None
        self.report_id = None
        self.report_generated_at = None
        self.medical_report_status_text = "Study selected. Run analysis to prepare a report."
        self.image_quality_report = None
        self.gradcam_image = None
        self.generated_pdf_path = None
        self.localized_image = None
        self._update_fracture_type_status()
        self._update_fracture_localization()
        self.gradcam_btn.configure(state="disabled")
        self.localization_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.generate_pdf_btn.configure(state="disabled")
        self.save_pdf_btn.configure(state="disabled")
        self._update_medical_report_page()
        try:
            with Image.open(filename) as img:
                self._set_preview(img.convert("RGB"))
            self._prepare_analysis_image()
            self._refresh_comparison_page()
        except (OSError, ValueError) as error:
            filename = ""
            self._set_status(f"Unable to preview image: {error}")

    def predict_gui(self):
        global filename
        if not filename:
            self._set_status("Upload an X-ray before running analysis")
            return

        if not self._prepare_analysis_image():
            return

        self.predict_btn.configure(state="disabled", text="Analyzing...")
        self.gradcam_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self._set_status("Analyzing X-ray with ResNet50 models...")
        try:
            bone_type_result, bone_confidence = predict(self.analysis_filename, "Parts", return_confidence=True)
            result, confidence = predict(self.analysis_filename, bone_type_result, return_confidence=True)
        except (OSError, ValueError, RuntimeError, tf.errors.InvalidArgumentError) as error:
            print(f"Prediction failed: {error}")
            self._set_status(f"Prediction failed: {error}")
            self.predict_btn.configure(state="normal", text="Run Analysis")
            return

        self.predicted_bone_type = bone_type_result
        self.current_result = result
        self.predicted_bone_confidence = bone_confidence
        self.fracture_confidence = confidence
        self.prediction_id = f"BS-{uuid.uuid4().hex[:12].upper()}"
        self.report_id = None
        self.report_generated_at = None
        self.medical_report_status_text = "Analysis complete. Generate the PDF report when ready."
        self.generated_pdf_path = None
        self.gradcam_image = None
        self.localized_image = None
        self._update_fracture_type_status()
        self._update_fracture_localization()
        self.predict_btn.configure(state="normal", text="Run Analysis")
        self.gradcam_btn.configure(state="normal")
        self.localization_btn.configure(state="normal" if result == "fractured" else "disabled")
        self.save_btn.configure(state="normal")
        self.generate_pdf_btn.configure(state="normal")
        self.save_pdf_btn.configure(state="disabled")
        self.res1_label.configure(text=f"Body part  {bone_type_result}")
        self.res1_conf_label.configure(text=f"Part confidence  {bone_confidence:.2f}%")
        self.res2_label.configure(text=f"Fracture status  {result.capitalize()}")
        self.res2_conf_label.configure(text=f"Fracture confidence  {confidence:.2f}%")
        if result == "fractured":
            self.result_badge.configure(text="  FRACTURE DETECTED  ", text_color=COLORS["red"], fg_color=COLORS["red_light"])
            self.fracture_progress.configure(progress_color=COLORS["red"], fg_color=COLORS["red_light"])
        else:
            self.result_badge.configure(text="  NO FRACTURE DETECTED  ", text_color=COLORS["green"], fg_color=COLORS["green_light"])
            self.fracture_progress.configure(progress_color=COLORS["green"], fg_color=COLORS["green_light"])
        self._animate_progress(self.part_progress, bone_confidence / 100)
        self._animate_progress(self.fracture_progress, confidence / 100)
        self.prediction_history.append((datetime.now().strftime("%H:%M"), bone_type_result, result, confidence))
        self.images_stat.configure(text=str(len(self.prediction_history)))
        self.confidence_stat.configure(text=f"{confidence:.1f}%")
        self._refresh_history()
        self._refresh_history_page()
        self._set_status("Analysis complete. Grad-CAM is ready.")
        self._update_medical_report_page()

    def _animate_progress(self, progress_bar, target):
        progress_bar.set(0)
        self._animate_step(progress_bar, target, 0)

    def _animate_step(self, progress_bar, target, current):
        current = min(target, current + 0.04)
        progress_bar.set(current)
        if current < target:
            self.after(18, lambda: self._animate_step(progress_bar, target, current))

    def _refresh_history(self):
        entries = self.prediction_history[-3:]
        text = "    ".join(f"{time} | {part} | {result.capitalize()} | {confidence:.1f}%" for time, part, result, confidence in entries)
        self.history_label.configure(text=text)

    def _set_status(self, message):
        self.save_label.configure(text=message, text_color=COLORS["muted"])

    def show_gradcam(self):
        global filename
        if not self.analysis_filename or not self.predicted_bone_type:
            self._refresh_gradcam_page()
            self._show_page("gradcam")
            return
        try:
            gradcam_image = generate_gradcam(self.analysis_filename, self.predicted_bone_type)
        except (OSError, ValueError, tf.errors.InvalidArgumentError) as error:
            self._set_status(f"Grad-CAM unavailable: {error}")
            return
        self.gradcam_image = gradcam_image
        self._refresh_gradcam_page()
        self._show_page("gradcam")

    def generate_pdf_report(self):
        if not self.prediction_id or not self.analysis_filename or not self.image_quality_report:
            self._set_status("Run analysis before generating a PDF report")
            self.medical_report_status_text = "Run analysis before generating a PDF report."
            self._update_medical_report_page()
            return

        self.generate_pdf_btn.configure(state="disabled", text="Generating PDF...")
        self.medical_report_status_text = "Generating the PDF report..."
        self._update_medical_report_page()
        try:
            if self.gradcam_image is None:
                self.gradcam_image = generate_gradcam(
                    self.analysis_filename,
                    self.predicted_bone_type,
                )
            if self.current_result == "fractured" and self.localized_image is None:
                self._update_fracture_localization()

            self.report_id = f"BSR-{uuid.uuid4().hex[:12].upper()}"
            generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_generated_at = generated_at
            fracture_type = (
                self.fracture_type_prediction
                if self.current_result == "fractured" and self.fracture_type_prediction
                else "Not Applicable"
                if self.current_result == "normal"
                else "Unavailable"
            )
            fracture_type_confidence = (
                f"{self.fracture_type_confidence:.2f}%"
                if self.current_result == "fractured" and self.fracture_type_confidence is not None
                else "--"
            )
            summary = (
                f"The body-part model predicted {self.predicted_bone_type} "
                f"({self.predicted_bone_confidence:.2f}% confidence). The fracture model returned "
                f"{self.current_result} ({self.fracture_confidence:.2f}% confidence). "
                + (
                    f"The fracture-type model predicted {fracture_type} "
                    f"({fracture_type_confidence} confidence)."
                    if self.current_result == "fractured" and self.fracture_type_prediction
                    else "Fracture type is not applicable to a normal result."
                    if self.current_result == "normal"
                    else "Fracture-type prediction is unavailable for this result."
                )
            )
            self.generated_pdf_path = os.path.join(
                tempfile.gettempdir(),
                f"bonesight_report_{self.report_id}.pdf",
            )
            generate_medical_report(
                self.generated_pdf_path,
                {
                    "generated_at": generated_at,
                    "report_id": self.report_id,
                    "prediction_id": self.prediction_id,
                    "study_name": os.path.basename(filename),
                    "body_part": self.predicted_bone_type,
                    "body_part_confidence": f"{self.predicted_bone_confidence:.2f}%",
                    "fracture_status": str(self.current_result).capitalize(),
                    "fracture_confidence": f"{self.fracture_confidence:.2f}%",
                    "fracture_type": fracture_type,
                    "fracture_type_confidence": fracture_type_confidence,
                    "image_quality": self.image_quality_report,
                    "original_image": filename,
                    "enhanced_image": self.enhanced_filename,
                    "gradcam_image": self.gradcam_image,
                    "localization_image": self.localized_image,
                    "summary": summary,
                },
            )
            self.medical_report_status_text = (
                f"PDF report generated at {self.report_generated_at} and ready to save."
            )
            self.save_pdf_btn.configure(state="normal")
            self._set_status(f"PDF report {self.report_id} generated. Choose Save PDF Report to export it.")
        except Exception as error:
            self.generated_pdf_path = None
            self.report_id = None
            self.report_generated_at = None
            self.medical_report_status_text = f"PDF report could not be generated: {error}"
            self.save_pdf_btn.configure(state="disabled")
            self._set_status(f"PDF report could not be generated: {error}")
        finally:
            self.generate_pdf_btn.configure(state="normal", text="Generate PDF Report")
            self._update_medical_report_page()

    def save_pdf_report(self):
        if not self.generated_pdf_path or not os.path.isfile(self.generated_pdf_path):
            self._set_status("Generate a PDF report before saving it")
            return
        destination = filedialog.asksaveasfilename(
            parent=self,
            initialdir=os.path.join(project_folder, "PredictResults"),
            title="Save BoneSight PDF report",
            initialfile=f"{self.report_id}.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF report", "*.pdf")],
        )
        if not destination:
            return
        try:
            shutil.copy2(self.generated_pdf_path, destination)
        except OSError as error:
            self.medical_report_status_text = f"PDF report could not be saved: {error}"
            self._update_medical_report_page()
            self._set_status(f"PDF report could not be saved: {error}")
            return
        self.medical_report_status_text = f"PDF report saved: {destination}"
        self._update_medical_report_page()
        self._set_status(f"PDF report saved to {destination}")

    def show_fracture_localization(self):
        if self.current_result != "fractured" or not self.analysis_filename or not self.predicted_bone_type:
            return
        try:
            self._update_fracture_localization()
            if self.localized_image is None:
                return
            with Image.open(filename) as source_image:
                original_image = source_image.convert("RGB")
            localized_image = self.localized_image
        except Exception as error:
            self._set_status(f"Fracture localization unavailable: {error}")
            return

        window = ctk.CTkToplevel(self)
        window.title("BoneSight | Fracture Localization")
        window.geometry("1000x720")
        window.configure(fg_color=COLORS["page"])
        window.transient(self)
        ctk.CTkLabel(window, text="Suspected fracture localization", text_color=COLORS["ink"], font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=28, pady=(22, 0))
        ctk.CTkLabel(window, text="The box marks a high-activation region from the fracture model; it is not a confirmed fracture boundary.", text_color=COLORS["muted"], font=ctk.CTkFont(size=11)).pack(anchor="w", padx=28, pady=(3, 16))
        image_frame = ctk.CTkFrame(window, fg_color="transparent")
        image_frame.pack(fill="both", expand=True, padx=22, pady=(0, 22))
        original_display = original_image.copy()
        original_display.thumbnail((440, 520), Image.Resampling.LANCZOS)
        localized_display = localized_image.copy()
        localized_display.thumbnail((440, 520), Image.Resampling.LANCZOS)
        self._add_explanation_image(image_frame, "Original X-ray", original_display, 0)
        self._add_explanation_image(image_frame, "Suspected region", localized_display, 1)

    def _add_explanation_image(self, parent, title, img, column):
        card = ctk.CTkFrame(parent, fg_color=COLORS["white"], corner_radius=12, border_width=1, border_color=COLORS["line"])
        card.grid(row=0, column=column, sticky="nsew", padx=6)
        parent.grid_columnconfigure(column, weight=1)
        ctk.CTkLabel(card, text=title, text_color=COLORS["ink"], font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(14, 4))
        photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        label = ctk.CTkLabel(card, text="", image=photo)
        label.pack(expand=True, padx=12, pady=(0, 14))
        setattr(label, "image", photo)

    def save_result(self):
        tempdir = filedialog.asksaveasfilename(parent=self, initialdir=project_folder + '/PredictResults/', title='Please select a directory and filename', defaultextension=".png")
        if not tempdir:
            return
        window = pygetwindow.getWindowsWithTitle('Bone Fracture Detection')[0]
        left, top = window.topleft
        right, bottom = window.bottomright
        pyautogui.screenshot(tempdir)
        im = Image.open(tempdir)
        im = im.crop((left + 10, top + 35, right - 10, bottom - 10))
        im.save(tempdir)
        self.save_label.configure(text="Saved to selected location", text_color=COLORS["green"])

    def open_image_window(self):
        im = Image.open(folder_path + "rules.jpeg")
        im = im.resize((700, 700))
        im.show()


if __name__ == "__main__":
    app = App()
    app.mainloop()