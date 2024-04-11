from msilib.schema import CheckBox
from PyQt5 import QtWidgets, QtCore, QtGui
from checkRelationship import extract_high_mi_features
import os

class ToggleButton(QtWidgets.QPushButton):
    def __init__(self, dialog, checkboxes, parent=None):
        super(ToggleButton, self).__init__(parent)
        self.dialog = dialog
        self.checkboxes = checkboxes
        self.setCheckable(True)
        self.setChecked(False)
        self.setText("OFF")
        self.setStyleSheet("QPushButton { background-color: red; }")
        self.toggled.connect(self.onToggled)

    def onToggled(self, checked):
        if checked:
            self.setText("ON")
            self.setStyleSheet("QPushButton { background-color: green; }")
            # Uncheck all checkboxes and clear the selected_data_columns list
            for checkbox in self.checkboxes:
                checkbox.setChecked(False)
                checkbox.setEnabled(False)
            self.dialog.clear_selected_data_columns()
        else:
            self.setText("OFF")
            self.setStyleSheet("QPushButton { background-color: red; }")
            # Set all checkboxes to be clickable
            for checkbox in self.checkboxes:
                checkbox.setEnabled(True)

class SelectAnalyseTargetDialog(QtWidgets.QDialog):
    def __init__(self, column_names, data_path, parent=None):
        super(SelectAnalyseTargetDialog, self).__init__(parent)
        self.data_path = data_path  # Add the dataframe as an attribute
        self.setWindowTitle("Please Select  Data & Target")
        self.setStyleSheet("background-color: #301860;")
        
        self.icon_path = os.path.join(os.path.dirname(__file__), '../media/icon.png')
        if not os.path.exists(self.icon_path):
            print(f"Icon file not found at path: {self.icon_path}")
        else:
            self.setWindowIcon(QtGui.QIcon(self.icon_path))
        
        self.column_names = column_names
        self.selected_data_columns = []
        self.selected_target_column = None
            
        # Main layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignCenter)   
        
        # Header
        self.header = QtWidgets.QLabel("Please Select Data & Target")
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

        # Label for Select target Data
        self.info_label = QtWidgets.QLabel(" Select target Data:")
        self.info_label.setFixedHeight(20)
        self.info_label.setStyleSheet("background-color: #d9abff;")
        self.layout.addWidget(self.info_label)
        
        # Create a grid layout for Checkboxes, Dropdown and button
        self.grid_layout = QtWidgets.QGridLayout()
        self.layout.addLayout(self.grid_layout)
        
        # Checkboxes for selecting multiple data columns
        self.checkboxes = []
        for i, column in enumerate(self.column_names):
            checkbox = QtWidgets.QCheckBox(column)
            checkbox.setStyleSheet("color: #ffffff;")
            checkbox.stateChanged.connect(self.checkbox_state_changed)
            self.checkboxes.append(checkbox)
            self.grid_layout.addWidget(checkbox, i // 3, i % 3)  # Adjust the numbers for grid layout
            
        self.info_label = QtWidgets.QLabel(" Auto select of target Data:")
        self.info_label.setFixedHeight(20)
        self.info_label.setStyleSheet("background-color: #d9abff;")
        self.layout.addWidget(self.info_label)
        
        # Toggle button
        self.toggle_button = ToggleButton(self, self.checkboxes, self)
        self.layout.addWidget(self.toggle_button)
        
        # Dropdown for selecting the target column
        self.target_dropdown = QtWidgets.QComboBox()
        self.target_dropdown.addItems(self.column_names)
        self.target_dropdown.currentIndexChanged.connect(self.target_selection_changed)
        self.target_dropdown.setStyleSheet("background-color: #ffffff;")
        
        target_layout = QtWidgets.QHBoxLayout()

        # Create QLabel and set its style
        label = QtWidgets.QLabel(' Select target column:')
        label.setStyleSheet("background-color: #d9abff;")
        target_layout.addWidget(label)

        target_layout.addWidget(self.target_dropdown)
        self.layout.addLayout(target_layout)

        # OK button
        buttons_layout = QtWidgets.QHBoxLayout()
        ok_button = QtWidgets.QPushButton('OK')
        ok_button.clicked.connect(self.on_ok_button_clicked)
        ok_button.setFixedSize(200, 30)  # Set fixed size: width from button_size, height to 30
        ok_button.setStyleSheet("background-color: #cd67cd;\n"
                                        "border-style:outset;\n"
                                        "border-width:2px;\n"
                                        "border-radius:15px;\n"
                                        "border-color:#ffffff;")
        
        buttons_layout.addWidget(ok_button)
        self.layout.addLayout(buttons_layout)
    
    def checkbox_state_changed(self, state):
        sender = self.sender()
        if sender.isChecked():
            self.selected_data_columns.append(sender.text())
        else:
            self.selected_data_columns.remove(sender.text())

    def target_selection_changed(self, index):
        self.selected_target_column = self.target_dropdown.itemText(index)

    def get_selected_columns(self):
        if not self.selected_target_column:
            # For the 'No target selected' warning
            msg_no_target = QtWidgets.QMessageBox(self)
            msg_no_target.setWindowTitle('No target selected')
            msg_no_target.setIcon(QtWidgets.QMessageBox.Warning)
            msg_no_target.setText('Please select a target column.')
            msg_no_target.setStyleSheet("color: #ffffff;")
            msg_no_target.exec_()
            return None, None
        if not self.selected_data_columns:
            # For the 'No data columns selected' warning
            msg_no_data_columns = QtWidgets.QMessageBox(self)
            msg_no_data_columns.setWindowTitle('No data columns selected')
            msg_no_data_columns.setIcon(QtWidgets.QMessageBox.Warning)
            msg_no_data_columns.setText('Please select one or more data columns.')
            msg_no_data_columns.setStyleSheet("color: #ffffff;")
            msg_no_data_columns.exec_()
            return None, None
        if self.selected_target_column in self.selected_data_columns:
            # For the 'Invalid selection' warning
            msg_invalid_selection = QtWidgets.QMessageBox(self)
            msg_invalid_selection.setWindowTitle('Invalid selection')
            msg_invalid_selection.setIcon(QtWidgets.QMessageBox.Warning)
            msg_invalid_selection.setText('Target column cannot be one of the data columns.')
            msg_invalid_selection.setStyleSheet("color: #ffffff;")
            msg_invalid_selection.exec_()
            return None, None
        return self.selected_data_columns, self.selected_target_column
    
    def clear_selected_data_columns(self):
        self.selected_data_columns.clear()
        
    def on_ok_button_clicked(self):
        if self.toggle_button.isChecked():
            # If toggle is ON, run the feature extraction
            try:
                high_mi_features = extract_high_mi_features(self.data_path, self.selected_target_column)
                # Update checkboxes based on the high MI features
                for checkbox in self.checkboxes:
                    if checkbox.text() in high_mi_features:
                        checkbox.setChecked(True)
            except ValueError as e:
                # Handle the error if the target column was not found
                error_dialog = QtWidgets.QMessageBox(self)
                error_dialog.setWindowTitle('Error')
                error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
                error_dialog.setText(str(e))
                error_dialog.exec_()
        else:
            selected_data, selected_target = self.get_selected_columns()
            if selected_data is not None and selected_target is not None:
                pass

        self.accept()
