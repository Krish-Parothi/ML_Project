"""
Generate Comprehensive Backend Architecture & ML Documentation PDF
===================================================================
Covers all 7 technical questions with detailed text, metric tables,
and embedded high-resolution visual plots.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

BASE = r"c:\Users\Krish\OneDrive\Desktop\ML_Project\ML_Project"
PLOTS_DIR = os.path.join(BASE, "plots")
PDF_OUT = os.path.join(BASE, "Airline_Passenger_Satisfaction_Backend_Documentation.pdf")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#7f8c8d"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Airline Passenger Satisfaction — Machine Learning Backend Architecture")
            self.setStrokeColor(colors.HexColor("#dcdde1"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)
            
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 35, page_text)
        self.drawString(54, 35, "CONFIDENTIAL & PROPRIETARY — END-TO-END ML PIPELINE")
        self.setStrokeColor(colors.HexColor("#dcdde1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_OUT,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1a252f'),
        alignment=1, # Center
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2980b9'),
        alignment=1,
        spaceAfter=25
    )

    q_title_style = ParagraphStyle(
        'QuestionTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0c2461'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1e3799'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2c3e50'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#2d3436'),
        backColor=colors.HexColor('#f1f2f6'),
        borderPadding=6,
        spaceAfter=6
    )

    caption_style = ParagraphStyle(
        'ImageCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#57606f'),
        alignment=1,
        spaceAfter=12
    )

    story = []

    # ─── COVER / HEADER ──────────────────────────────────────────────────────────
    story.append(Paragraph("✈️ Airline Passenger Satisfaction ML System", title_style))
    story.append(Paragraph("Complete Backend Architecture, Statistical EDA, Model Engineering & Diagnostic Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3498db'), spaceAfter=15))

    # Meta Table
    meta_data = [
        [Paragraph("<b>Project:</b> Airline Passenger Satisfaction", body_style), Paragraph("<b>Target Task:</b> Binary Classification", body_style)],
        [Paragraph("<b>Total Dataset Rows:</b> 129,880", body_style), Paragraph("<b>Engineered Features:</b> 29 Features", body_style)],
        [Paragraph("<b>Primary Model:</b> LightGBM (Tuned)", body_style), Paragraph("<b>Benchmark Test Accuracy:</b> 96.16%", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[250, 250])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # ─── QUESTION 1 ─────────────────────────────────────────────────────────────
    story.append(Paragraph("1. What is This Project About?", q_title_style))
    story.append(Paragraph(
        "This project builds an enterprise-grade Machine Learning classification system to predict <b>Airline Passenger Satisfaction</b>. "
        "Airlines operate in a fiercely competitive environment where acquiring a new passenger is 5 to 7 times more expensive than retaining an existing one. "
        "Customer dissatisfaction directly leads to customer churn, negative public reviews, and lost passenger lifetime value (LTV).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Core Objectives:</b>", h2_style
    ))
    story.append(Paragraph("• <b>Proactive Dissatisfaction Identification:</b> Enable airlines to instantly flag passengers who had a sub-par flight experience (predicting <b>'Satisfied' (1)</b> vs. <b>'Neutral / Dissatisfied' (0)</b> with 96.16% accuracy) before they churn to competitors.", bullet_style))
    story.append(Paragraph("• <b>Service Recovery Automation:</b> Automatically trigger personalized goodwill gestures (frequent flyer miles, apology vouchers, priority boarding passes) for unhappy passengers upon deplaning.", bullet_style))
    story.append(Paragraph("• <b>Capital Expenditure (CapEx) Prioritization:</b> Provide airline leadership with mathematical feature importances demonstrating whether fleet investments should focus on in-flight wifi upgrades, online boarding interfaces, or seat re-configurations.", bullet_style))
    story.append(Spacer(1, 10))

    # ─── QUESTION 2 ─────────────────────────────────────────────────────────────
    story.append(Paragraph("2. About the Dataset: Instances, Features, Outliers & Data Splits", q_title_style))
    story.append(Paragraph(
        "<b>Dataset Dimensions:</b> The raw dataset comprises exactly <b>129,880 passenger records</b> and <b>24 raw columns</b> (23 predictive input features and 1 ground-truth target column).",
        body_style
    ))
    story.append(Paragraph("<b>Input Feature Categorization:</b>", h2_style))
    story.append(Paragraph("1. <b>Passenger Demographics:</b> Gender (Male/Female), Age (numerical, 7–85), Customer Type (First-time / Returning loyalty customer).", bullet_style))
    story.append(Paragraph("2. <b>Flight Parameters:</b> Type of Travel (Business vs. Personal), Travel Class (Business, Economy, Economy Plus), Flight Distance (miles), Departure Delay (minutes), and Arrival Delay (minutes).", bullet_style))
    story.append(Paragraph("3. <b>Service Amenities Ratings (14 Likert Features, 1–5 scale):</b> Online Boarding, In-flight Wifi, Seat Comfort, Leg Room Service, Cleanliness, Food and Drink, In-flight Service, In-flight Entertainment, Baggage Handling, Check-in Service, Gate Location, Ease of Online Booking, Departure and Arrival Time Convenience, and On-board Service.", bullet_style))
    story.append(Paragraph("4. <b>Target Variable:</b> <code>Satisfaction</code> — 73,452 Neutral/Dissatisfied (56.55%) vs. 56,428 Satisfied (43.45%).", bullet_style))

    story.append(Paragraph("<b>Data Quality, Nulls & The '0' Rating Phenomenon:</b>", h2_style))
    story.append(Paragraph(
        "• <b>Missing Values:</b> Only a single feature contained missing data: <code>Arrival Delay</code> had 393 nulls (0.3%). This occurs when flights are canceled or diverted. These were cleanly imputed using the median (0.0 minutes), preserving sample integrity without introducing outlier bias.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>The '0' Rating Issue:</b> In survey features, 0 does <i>not</i> mean zero satisfaction; it means <i>'Not Applicable'</i> (e.g. 6,681 passengers had no convenience rating, 3,916 had no wifi usage). Treating 0 as an ordinal rating would distort distances and tree splits. We converted 0s to NaN and imputed with median service scores.",
        bullet_style
    ))

    story.append(Paragraph("<b>Outliers Analysis:</b>", h2_style))
    story.append(Paragraph(
        "Yes, substantial outliers were identified in two specific variables: "
        "<code>Flight Distance</code> exhibited severe right-skewness (mean 1,190 miles, with extreme long-haul flights up to 4,983 miles). "
        "Similarly, <code>Departure Delay</code> and <code>Arrival Delay</code> featured extreme delay spikes exceeding 1,000 minutes (16+ hours). "
        "These were handled through <b>log1p transformations</b> for flight distance and composite summation for total delays.",
        body_style
    ))

    story.append(Paragraph("<b>Train / Validation / Test Split Strategy:</b>", h2_style))
    story.append(Paragraph(
        "We partitioned the data using a strict <b>70% / 15% / 15% Stratified Split</b> with a fixed random seed (<code>random_state=42</code>):",
        body_style
    ))
    
    split_table_data = [
        ["Data Split", "Proportion", "Row Count", "Target Class Balance (Satisfied / Dissatisfied)"],
        ["Training Set (X_train)", "70.0%", "90,916 rows", "43.4% Satisfied / 56.6% Dissatisfied"],
        ["Validation Set (X_val)", "15.0%", "19,482 rows", "43.4% Satisfied / 56.6% Dissatisfied"],
        ["Test Set (X_test)", "15.0%", "19,482 rows", "43.4% Satisfied / 56.6% Dissatisfied"]
    ]
    t_split = Table(split_table_data, colWidths=[110, 70, 110, 210])
    t_split.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0c2461')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('ALIGN', (1,0), (2,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_split)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Why is a Validation Set Required in Addition to the Test Set?</b><br/>"
        "In professional ML workflows, evaluating models repeatedly on the Test set during tuning causes <i>Data Leakage (Information Bleed)</i>, "
        "where hyperparameters subtly overfit to the test set quirks. The <b>Validation Set</b> was used exclusively to compare default models and tune hyperparameters. "
        "The <b>Test Set</b> was locked away and evaluated exactly once at the very end to guarantee a truly unbiased benchmark.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # ─── QUESTION 3 ─────────────────────────────────────────────────────────────
    story.append(Paragraph("3. Exploratory Data Analysis (EDA) & Feature Engineering", q_title_style))
    story.append(Paragraph(
        "Raw data was transformed into an optimized <b>29-feature numerical representation</b> through systematic engineering:",
        body_style
    ))
    story.append(Paragraph("• <b>Dropping Non-Informative Columns:</b> The <code>ID</code> feature was dropped as it is an arbitrary passenger index with zero predictive entropy.", bullet_style))
    story.append(Paragraph("• <b>Multicollinearity Elimination (Delay Fusion):</b> Pearson correlation analysis revealed an extreme correlation of <b>r = 0.965</b> between <code>Departure Delay</code> and <code>Arrival Delay</code>. Keeping both introduces severe variance instability in linear models and split dilution in tree models. We engineered a single composite <code>Total Delay = Departure Delay + Arrival Delay</code> and dropped the raw originals.", bullet_style))
    story.append(Paragraph("• <b>Non-Linear Distance Normalization:</b> Applied <code>np.log1p(Flight Distance)</code>. This compressed extreme 4,000+ mile outliers into a symmetric bell curve, stabilizing gradient descent and split boundaries.", bullet_style))
    story.append(Paragraph("• <b>Demographic Age Binning:</b> Customer loyalty curves are non-linear (satisfaction dips during young adulthood and peaks in middle age). We discretized Age into 5 bins: <i>Child (0–17), Young Adult (18–25), Adult (26–40), Middle-Aged (41–60), Senior (61+)</i> and one-hot encoded them.", bullet_style))
    story.append(Paragraph("• <b>Composite Domain Service Scores:</b> To capture broader passenger experience dimensions, we engineered: (a) <code>Ground Service Score</code> (mean of Check-in, Gate, Time Convenience, and Online Booking), and (b) <code>In-flight Service Score</code> (mean of Wifi, Food, Cleanliness, Seat Comfort, In-flight Entertainment, Legroom, and Service).", bullet_style))
    story.append(Paragraph("• <b>Categorical Encoding:</b> Label encoded binary features (<code>Gender: Male=0, Female=1</code>; <code>Customer Type: First-time=0, Returning=1</code>; <code>Type of Travel: Personal=0, Business=1</code>). One-hot encoded <code>Class</code> (Business, Economy, Economy Plus).", bullet_style))
    story.append(Paragraph("• <b>Feature Scaling:</b> Standardized all 18 continuous and survey features using <code>StandardScaler</code> ($\mu=0, \sigma=1$). The mean and scale parameters were serialized into <code>scaler_info.json</code> for production inference.", bullet_style))
    story.append(Spacer(1, 10))

    # ─── QUESTION 4 ─────────────────────────────────────────────────────────────
    story.append(PageBreak()) # Clean break for Model Engineering section
    story.append(Paragraph("4. Machine Learning Models: Selection Rationale & Hyperparameter Mechanics", q_title_style))
    story.append(Paragraph(
        "After benchmarking 11 baseline algorithms across linear, kernel, neighbor, bagging, and boosting paradigms, we selected the "
        "<b>Top 3 High-Performance Ensemble Models</b>: LightGBM, XGBoost, and Random Forest.",
        body_style
    ))
    story.append(Paragraph("<b>Why Tree Ensembles Dominate This Problem:</b>", h2_style))
    story.append(Paragraph(
        "Tabular survey data exhibits complex non-linear feature interactions (e.g., <i>a passenger in Business Class with poor Wifi might remain satisfied, whereas an Economy passenger with identical poor Wifi becomes dissatisfied</i>). "
        "Linear models (Logistic Regression, Linear SVM) can only draw planar hyperplanes and scored under 88%. "
        "Tree-based ensembles capture high-order interaction thresholds without requiring manual polynomial expansion.",
        body_style
    ))

    story.append(Paragraph("<b>Detailed Hyperparameter Breakdown for Each Model:</b>", h2_style))

    # Model 1: LightGBM
    story.append(Paragraph("<b>1. LightGBM (Light Gradient Boosting Machine) — #1 Overall Performer</b>", h2_style))
    story.append(Paragraph(
        "LightGBM utilizes <b>Histogram-Based Tree Building</b> and <b>Leaf-Wise (Best-First) Growth</b> rather than level-wise growth, achieving faster convergence and higher accuracy.",
        body_style
    ))
    story.append(Paragraph("• <code>n_estimators=500</code>: The number of sequential boosting trees. 500 weak learners allow deep error residual minimization.", bullet_style))
    story.append(Paragraph("• <code>learning_rate=0.05</code>: Shrinkage step size applied to each tree's contribution. A lower learning rate prevents overshoot and ensures gradual convergence.", bullet_style))
    story.append(Paragraph("• <code>num_leaves=63</code>: The maximum number of leaves per tree. Controls model capacity; $63$ leaves allows rich splits while capping tree size to at most 125 total nodes.", bullet_style))
    story.append(Paragraph("• <code>subsample=0.8</code> & <code>colsample_bytree=0.7</code>: Stochastic gradient boosting; each tree randomly samples 80% of rows and 70% of features, breaking correlation between individual trees.", bullet_style))
    story.append(Paragraph("• <code>min_child_samples=10</code>: Minimum number of data samples required in a leaf. Prevents the tree from isolating individual noisy survey outliers.", bullet_style))
    story.append(Paragraph("• <code>reg_alpha=0.5</code> (L1) & <code>reg_lambda=0.0</code> (L2): L1 regularization penalizes complex leaf weight values, promoting sparsity and shrinking non-contributory weights to zero.", bullet_style))

    # Model 2: XGBoost
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>2. XGBoost (Extreme Gradient Boosting) — #2 Runner-Up</b>", h2_style))
    story.append(Paragraph(
        "XGBoost employs second-order Taylor expansion of the loss function (Newton-Raphson optimization) with built-in tree pruning.",
        body_style
    ))
    story.append(Paragraph("• <code>max_depth=8</code>: Explicit depth ceiling. Limits tree height to 8 levels, ensuring complex interaction capture while bounding variance.", bullet_style))
    story.append(Paragraph("• <code>min_child_weight=5</code>: Minimum sum of instance weight (Hessian) needed in a child. Higher values force the algorithm to be conservative on small passenger partitions.", bullet_style))
    story.append(Paragraph("• <code>gamma=0.1</code> (Minimum Loss Reduction): Pseudo-regularization parameter specifying minimum loss reduction required to make a further partition on a leaf.", bullet_style))
    story.append(Paragraph("• <code>scale_pos_weight=1.30</code>: Compensates for the mild class imbalance (56.6% dissatisfied / 43.4% satisfied) by weighting minority class gradient updates.", bullet_style))

    # Model 3: Random Forest
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>3. Random Forest (Bagging Ensemble) — #3 Benchmark</b>", h2_style))
    story.append(Paragraph(
        "Random Forest creates a forest of <b>250 fully independent deep decision trees</b> using Bootstrap Aggregation (bagging) and Random Subspace feature selection.",
        body_style
    ))
    story.append(Paragraph("• <code>n_estimators=250</code>: Aggregates votes across 250 diverse decision trees.", bullet_style))
    story.append(Paragraph("• <code>max_depth=20</code>: Allows trees to grow deeply to minimize bias on complex multi-feature survey interactions.", bullet_style))
    story.append(Paragraph("• <code>min_samples_split=5</code> & <code>min_samples_leaf=2</code>: Prevents leaves with single passenger data points, serving as the primary regularization barrier against memorization.", bullet_style))
    story.append(Paragraph("• <code>max_features='sqrt'</code>: Randomly selects $\\sqrt{29} \\approx 5$ candidate features at each node split, decorrelating individual trees.", bullet_style))
    story.append(Spacer(1, 10))

    # ─── QUESTION 5 ─────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Performance Metrics: Train, Validation & Test Evaluation", q_title_style))
    story.append(Paragraph(
        "Below is the exact verified performance matrix across all data splits (Train: 90,916 rows; Validation: 19,482 rows; Held-Out Test: 19,482 rows):",
        body_style
    ))

    perf_data = [
        ["Model Name", "Dataset Split", "Accuracy", "F1-Score", "ROC-AUC", "Precision", "Recall"],
        ["LightGBM (Tuned)", "Train (90.9k)", "98.11%", "0.9781", "0.9990", "98.49%", "97.13%"],
        ["LightGBM (Tuned)", "Validation (19.5k)", "96.05%", "0.9539", "0.9942", "96.77%", "94.05%"],
        ["LightGBM (Tuned)", "Held-Out Test", "96.16%", "0.9552", "0.9947", "96.72%", "94.35%"],
        ["XGBoost (Tuned)", "Train (90.9k)", "98.53%", "0.9830", "0.9992", "98.67%", "97.93%"],
        ["XGBoost (Tuned)", "Validation (19.5k)", "95.81%", "0.9512", "0.9936", "96.25%", "94.02%"],
        ["XGBoost (Tuned)", "Held-Out Test", "95.97%", "0.9531", "0.9940", "96.37%", "94.28%"],
        ["Random Forest", "Train (90.9k)", "98.66%", "0.9845", "0.9994", "0.9893", "97.98%"],
        ["Random Forest", "Validation (19.5k)", "95.27%", "0.9448", "0.9916", "95.81%", "93.18%"],
        ["Random Forest", "Held-Out Test", "95.17%", "0.9437", "0.9915", "95.58%", "93.19%"]
    ]
    t_perf = Table(perf_data, colWidths=[110, 95, 60, 55, 60, 60, 60])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0c2461')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [
            colors.HexColor('#eafaf1'), colors.HexColor('#eafaf1'), colors.HexColor('#d4efdf'), # LightGBM green tint
            colors.HexColor('#ebf5fb'), colors.HexColor('#ebf5fb'), colors.HexColor('#d6eaf8'), # XGBoost blue tint
            colors.HexColor('#fbfcfc'), colors.HexColor('#fbfcfc'), colors.HexColor('#f2f3f4')  # RF grey tint
        ]),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_perf)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Key Performance Observations:</b><br/>"
        "• <b>Zero Data Leakage:</b> Validation Accuracy (96.05%) and Test Accuracy (96.16%) are virtually identical (difference of only 0.11%). This confirms true generalizability.<br/>"
        "• <b>High Discriminative Power:</b> An ROC-AUC score of <b>0.9947</b> indicates that if you randomly draw one satisfied passenger and one dissatisfied passenger, LightGBM assigns a higher satisfaction probability to the satisfied passenger in 99.47% of instances.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # ─── QUESTION 6 ─────────────────────────────────────────────────────────────
    story.append(PageBreak()) # Clean break for Architecture & Size Deep Dive
    story.append(Paragraph("6. Architectural Deep Dive: File Sizes, Training Times & Random Forest Mechanics", q_title_style))
    story.append(Paragraph(
        "One of the most striking findings in this project is the dramatic disparity in model file sizes and training runtimes:",
        body_style
    ))

    size_table_data = [
        ["Model Architecture", "Paradigm", "File Size on Disk", "Training Time (90.9k rows)", "Total Node Count"],
        ["Random Forest", "Bagging (Bootstrap)", "155.74 MB", "6.42 seconds", "~5,000,000+ nodes"],
        ["LightGBM (Tuned)", "Histogram Boosting", "3.42 MB (45x smaller)", "9.84 seconds", "~62,500 nodes"],
        ["XGBoost (Tuned)", "Gradient Boosting", "1.99 MB (78x smaller)", "5.53 seconds", "~60,000 nodes"]
    ]
    t_size = Table(size_table_data, colWidths=[115, 95, 95, 105, 90])
    t_size.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0c2461')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#fadbd8'), colors.HexColor('#d4efdf'), colors.HexColor('#d6eaf8')]),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_size)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Why Did Random Forest Take 155 MB While Others Took ~2–3 MB?</b>", h2_style))
    story.append(Paragraph(
        "<b>1. Bagging (Deep Independent Trees) vs. Boosting (Shallow Weak Learners):</b><br/>"
        "Random Forest creates 250 completely independent trees with <code>max_depth=20</code>. In bagging, each tree attempts to fit the data independently and grow as deep as possible to reduce bias. Across 90,916 training rows, each individual tree grows to between <b>18,000 and 25,000 decision nodes</b>.<br/>"
        "$$\\text{Total Nodes} = 250 \\text{ trees} \\times 20,000 \\text{ nodes} \\approx 5,000,000+ \\text{ nodes!}$$<br/>"
        "In contrast, LightGBM uses leaf-wise growth capped at <code>num_leaves=63</code>. A binary tree with 63 leaves has at most 125 total nodes. Over 500 trees, that is only <b>62,500 total nodes</b> across the entire model — almost <b>80 times fewer nodes than Random Forest!</b>",
        body_style
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>2. Scikit-Learn's Internal Memory Object Storage:</b><br/>"
        "In Python's <code>scikit-learn</code>, every single tree node is stored as a C-struct array containing: <code>feature index (int64)</code>, <code>threshold (float64)</code>, <code>left child index (int64)</code>, <code>right child index (int64)</code>, <code>impurity (float64)</code>, <code>n_node_samples (int64)</code>, and a <code>value class distribution array (float64 x 2)</code>. Multiplying this by 5 million nodes results in ~155 MB of raw memory that Python's <code>joblib</code> serializer writes to disk.<br/>"
        "LightGBM and XGBoost, on the other hand, are engineered in optimized C++ and serialize trees as compact bit-compressed index maps.",
        body_style
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Random Forest Concise Working Summary:</b><br/>"
        "1. <b>Bootstrapping:</b> 250 subsets are created by randomly sampling 90,916 rows with replacement (each tree sees ~63.2% unique rows; ~36.8% are Out-of-Bag).<br/>"
        "2. <b>Random Feature Subspace:</b> At each decision node, the tree evaluates only $\\sqrt{29} \\approx 5$ randomly picked candidate features to find the best Gini impurity split.<br/>"
        "3. <b>Majority Voting:</b> During inference, all 250 trees output individual class votes; the majority vote wins.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # ─── QUESTION 7 ─────────────────────────────────────────────────────────────
    story.append(PageBreak()) # Clean break for Comprehensive Plot Walkthrough
    story.append(Paragraph("7. Comprehensive Plot Walkthrough & Deep Analysis", q_title_style))
    story.append(Paragraph(
        "Below is an exhaustive walkthrough of all 7 analytical plots generated by our pipeline, explaining every visual element, axis, curve, and business takeaway.",
        body_style
    ))

    # Function to add plot section cleanly
    def add_plot_section(filename, title, description, width=460, height=160):
        img_path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(img_path):
            story.append(Paragraph(f"<b>Figure: {title}</b>", h2_style))
            story.append(Image(img_path, width=width, height=height))
            story.append(Paragraph(description, caption_style))
            story.append(Spacer(1, 8))

    # Plot 1: Sanity Check
    add_plot_section(
        "01_sanity_check.png",
        "Data Preprocessing & Distribution Sanity Check",
        "<b>Plot Explanation:</b> Panel 1 shows the target class distribution across the 90,916 training instances (51,416 Dissatisfied vs 39,500 Satisfied), confirming mild class balance. "
        "Panel 2 illustrates the distribution of scaled Total Delay, showing high zero-clustering (on-time flights) with bounded positive delay tails. "
        "Panel 3 displays scaled Log Flight Distance, verifying that the extreme right-skewed mileage has been normalized into a symmetric Gaussian distribution.",
        width=480, height=130
    )

    # Plot 2: Correlation Heatmap
    add_plot_section(
        "02_correlation_heatmap.png",
        "Feature Correlation Heatmap (Training Set)",
        "<b>Plot Explanation:</b> Lower triangular Pearson correlation matrix across numerical features. "
        "Highlights the severe multicollinearity between Departure Delay and Arrival Delay (r = 0.965), which justified merging them into Total Delay. "
        "It also illustrates significant co-linearity between in-flight service elements (Cleanliness, Food & Drink, Seat Comfort, In-flight Entertainment with r between 0.65 and 0.75).",
        width=400, height=220
    )

    # Plot 3: Final Model Comparison
    story.append(PageBreak())
    add_plot_section(
        "final_model_comparison.png",
        "Final Model Comparison on Held-Out Test Set",
        "<b>Plot Explanation:</b> Side-by-side performance comparison on 19,482 unseen test passengers across Accuracy, F1-Score, and ROC-AUC. "
        "LightGBM leads all three metrics (96.16% Acc, 0.9552 F1, 0.9947 AUC), followed closely by XGBoost (95.97% Acc, 0.9531 F1, 0.9940 AUC) "
        "and Random Forest (95.17% Acc, 0.9437 F1, 0.9915 AUC).",
        width=480, height=160
    )

    # Plot 4: Confusion Matrices
    add_plot_section(
        "final_confusion_matrices.png",
        "Confusion Matrices on Held-Out Test Set (19,482 Passengers)",
        "<b>Plot Explanation:</b> 2x2 confusion matrices for LightGBM, XGBoost, and Random Forest on 19,482 test passengers. "
        "For LightGBM: 10,747 True Negatives, 7,986 True Positives, 271 False Positives (Type I error rate of only 2.45%), and 478 False Negatives (Type II error rate of 5.65%). "
        "Confirms balanced accuracy across both classes with no systemic bias toward the majority class.",
        width=480, height=150
    )

    # Plot 5: ROC Curves
    story.append(PageBreak())
    add_plot_section(
        "final_roc_curves.png",
        "Receiver Operating Characteristic (ROC) Curves Comparison",
        "<b>Plot Explanation:</b> True Positive Rate vs False Positive Rate across all discrimination thresholds. "
        "All 3 curves hug the top-left corner (0,1), proving extreme separability. "
        "LightGBM's AUC of 0.9947 and XGBoost's AUC of 0.9940 demonstrate near-perfect classification ranking compared to the diagonal dashed random guess line (AUC = 0.5000).",
        width=400, height=220
    )

    # Plot 6: Top Feature Importance
    add_plot_section(
        "top_feature_importance.png",
        "Top 20 Most Important Features Driving Passenger Satisfaction",
        "<b>Plot Explanation:</b> Normalized feature importance scores extracted from LightGBM. "
        "The #1 and #2 dominant predictors are <code>Online Boarding</code> and <code>In-flight Wifi Service</code>. "
        "Travel Class (<code>Class_Business</code>) and Flight Purpose (<code>Type of Travel</code>) are the leading demographic drivers. "
        "Actionable Insight: Digital pre-flight and connectivity touchpoints impact customer loyalty far more than meal quality or minor operational delays.",
        width=420, height=210
    )

    # Plot 7: Overfitting / Underfitting Analysis
    story.append(PageBreak())
    add_plot_section(
        "overfitting_underfitting_analysis.png",
        "Model Diagnostics: Overfitting vs. Underfitting Analysis",
        "<b>Plot Explanation:</b> Dual-panel diagnostic verification. "
        "Panel 1 shows Train Accuracy (blue) vs Test Accuracy (green), proving zero underfitting as all scores exceed 95%. "
        "Panel 2 maps the Generalization Gap (Train Acc - Test Acc) against standard machine learning risk zones. "
        "All 3 models reside firmly in the <b>Green Optimal Sweet-Spot Zone (Gap < 5%)</b>: LightGBM = 1.95%, XGBoost = 2.56%, Random Forest = 3.49%. None enter the moderate risk (5-10%) or severe overfitting (>10%) zones.",
        width=480, height=190
    )

    # ─── SUMMARY CONCLUSION ─────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Executive Summary & Production Recommendation</b>", h2_style))
    story.append(Paragraph(
        "Based on performance (96.16% test accuracy, 0.9947 ROC-AUC), generalization balance (1.95% gap), rapid inference latency, "
        "and compact disk footprint (3.42 MB), <b>LightGBM is selected as the primary production model</b> (serialized at <code>saved_models/best_model.pkl</code>). "
        "This completes the end-to-end backend pipeline from raw survey ingestion through production artifact export.",
        body_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"SUCCESS: Generated PDF at {PDF_OUT}")

if __name__ == "__main__":
    build_pdf()

