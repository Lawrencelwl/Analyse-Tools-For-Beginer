from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox
from hoverButton import HoverButton
from functools import partial
import os

class ModelSelectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(ModelSelectionDialog, self).__init__(parent)
        self.setWindowTitle("Please Select A Model")
        self.setFixedSize(420, 220)
        self.setStyleSheet("background-color: #301860;")
        
        self.icon_path = os.path.join(os.path.dirname(__file__), '../media/icon.png')
        if not os.path.exists(self.icon_path):
            print(f"Icon file not found at path: {self.icon_path}")
        else:
            self.setWindowIcon(QtGui.QIcon(self.icon_path))
        
        self.selected_model_name = None

        # Main layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignCenter)

        # Header
        self.header = QtWidgets.QLabel("Please Select A Model")
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.header.setFont(font)
        self.header.setAlignment(QtCore.Qt.AlignCenter)
        self.header.setStyleSheet("color: #ffffff;"
                                  "background-color: #604878;")
        self.header.setFixedSize(400,80)
        self.layout.addWidget(self.header)

        # Create a grid layout for buttons
        self.grid_layout = QtWidgets.QGridLayout()
        self.layout.addLayout(self.grid_layout)
        
        # Button positioning and size data
        button_positions = [(i, j) for i in range(5) for j in range(2)]
        button_size = QtCore.QSize(160, 50)

        # Store the currently clicked button
        self.currently_clicked_button = None
        
        model_name = ["Logistic Regression Model", "Random Forest Model"]  # Use a list to maintain order

        # Dynamically add buttons for each model
        for position, model_number in zip(button_positions, range(1, 3)):
            button = HoverButton(self)
            button.setText(model_name[model_number-1])
            button.clicked.connect(partial(self.set_selected_model, button))
            
            button.setFixedSize(button_size)
            self.grid_layout.addWidget(button, *position)

        # Confirm button layout
        confirm_layout = QtWidgets.QHBoxLayout()
        confirm_layout.addStretch()  # Add stretchable space on the left side

        self.confirmButton = HoverButton(self)
        self.confirmButton.setText("Confirm")
        self.confirmButton.setStyleSheet("background-color: #cd67cd;\n"
                                        "border-style:outset;\n"
                                        "border-width:2px;\n"
                                        "border-radius:15px;\n"
                                        "border-color:#ffffff;")
        self.confirmButton.setFixedSize(200, 30)  # Set fixed size: width from button_size, height to 30
        self.confirmButton.clicked.connect(self.accept)

        confirm_layout.addWidget(self.confirmButton)  # Add the button to the horizontal layout

        confirm_layout.addStretch()  # Add stretchable space on the right side

        # Add the confirm button layout to the main layout
        self.layout.addLayout(confirm_layout)
        
    def set_selected_model(self, button):
        if self.currently_clicked_button:
            self.currently_clicked_button.setDefaultStyle()
        button.setClickedStyle()
        self.currently_clicked_button = button
        self.selected_model_name = button.text()

    def selected_model(self):
        if self.selected_model_name is None:
            msg = QMessageBox()
            msg.setWindowTitle("Model Selection")
            msg.setText("Please select a model before confirming.")
            msg.setIcon(QMessageBox.Warning)
            msg.setStyleSheet("background-color: #301860;color: #ffffff;")
            msg.setWindowIcon(QtGui.QIcon(self.icon_path))
            msg.exec_()  # Display the message box
        else:
            return self.selected_model_name