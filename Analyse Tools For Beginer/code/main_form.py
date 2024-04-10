import os
import sys
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QThread
from hoverButton import HoverButton
from modelSelectionDialog import ModelSelectionDialog
from selectAnalyseTargetDialog import SelectAnalyseTargetDialog
from handler import HandlerWorker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class Ui_Form(object):
    def __init__(self):
        super().__init__()
        # Initialize the selected_file_path attribute
        self.selected_file_path = None
        self.selected_model = None
        self.handler_worker = HandlerWorker()
        self.handler_worker_thread = QThread()
        self.handler_worker.moveToThread(self.handler_worker_thread)
        self.handler_worker.finished.connect(self.on_handling_finished)
        # Connect the thread's started signal to the worker's handle_data method
        self.handler_worker_thread.started.connect(self.handler_worker.handle_data)
        # Connect data_ready signal to a new slot method
        self.handler_worker.data_ready.connect(self.on_data_ready)

    def setupUi(self, Form):
        self.Form = Form
        Form.setObjectName("Form")
        Form.resize(900, 570)
        Form.setStyleSheet("background-color: #301860;")
        self.header = QtWidgets.QWidget(Form)
        self.header.setGeometry(QtCore.QRect(0, 0, 900, 80))
        self.header.setStyleSheet("background-color: #604878;")
        self.header.setObjectName("header")
        self.label = QtWidgets.QLabel(self.header)
        self.label.setGeometry(QtCore.QRect(280, 15, 350, 50))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.label.setFont(font)
        self.label.setStyleSheet("color: #ffffff;\n"
                                "border-style:outset;\n"
                                "border-width:2px;\n"
                                "border-radius:15px;\n"
                                "border-color:#ffffff;")
        self.label.setObjectName("label")
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(0, 80, 900, 570))
        self.widget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.widget.setAutoFillBackground(False)
        self.widget.setStyleSheet("background-color: #301860;")
        self.widget.setObjectName("widget")
        self.Confirm = HoverButton(self.widget)
        self.Confirm.setGeometry(QtCore.QRect(700, 430, 160, 30))
        self.Confirm.setCheckable(True)
        self.Confirm.clicked.connect(self.start_analyse)
        self.Confirm.setObjectName("Confirm")
        self.tableWidget = QtWidgets.QTableWidget(self.widget)
        self.tableWidget.setGeometry(QtCore.QRect(30, 30, 620, 430))
        self.tableWidget.setStyleSheet("background-color: #000000;\n"
"border-style:outset;\n"
"border-width:2px;\n"
"border-radius:15px;")
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.SelectModel = HoverButton(self.widget)
        self.SelectModel.clicked.connect(self.select_model)
        self.SelectModel.setGeometry(QtCore.QRect(700, 130, 160, 30))
        self.SelectModel.setCheckable(True)
        self.SelectModel.setObjectName("SelectModel")
        self.SelectAnalyseTarget = HoverButton(self.widget)
        self.SelectAnalyseTarget.setGeometry(QtCore.QRect(700, 80, 160, 30))
        self.SelectAnalyseTarget.setCheckable(True)
        self.SelectAnalyseTarget.clicked.connect(self.select_analyse_target)
        self.SelectAnalyseTarget.setObjectName("SelectAnalyseTarget")
        self.SelectFile = HoverButton(self.widget)
        self.SelectFile.clicked.connect(self.select_file)
        self.SelectFile.setGeometry(QtCore.QRect(700, 30, 160, 30))
        self.SelectFile.setCheckable(True)
        self.SelectFile.setObjectName("SelectFile")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Analyse Tools For Beginer"))
        self.label.setText(_translate("Form", "Analyse Tools For Beginer"))
        self.Confirm.setText(_translate("Form", "Confirm"))
        self.SelectModel.setText(_translate("Form", "Select Model"))
        self.SelectAnalyseTarget.setText(_translate("Form", "Select Analyse Target"))
        self.SelectFile.setText(_translate("Form", "Select File"))

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv);;All Files (*.*)")
        if file_path:
            # Store the selected file path
            self.selected_file_path = file_path
            print(f"{self.selected_file_path}")
            try:
                df = pd.read_csv(file_path)
                self.load_data_into_table(df)
                msg = QMessageBox()
                msg.setWindowTitle("File Selected")
                msg.setText("The table will only show the first 50 records.")
                msg.setIcon(QMessageBox.Information)
                msg.setStyleSheet("background-color: #301860; color: #ffffff;")
                msg.setWindowIcon(QtGui.QIcon(icon_path))
                msg.exec_()
            except Exception as e:
                QMessageBox.critical(None, "Error", f"An error occurred while reading the file: {e}")
        return None

    def load_data_into_table(self, dataframe, limit=50):
        if len(dataframe) > limit:
            dataframe = dataframe.head(limit)
            
        self.tableWidget.setRowCount(dataframe.shape[0])
        self.tableWidget.setColumnCount(dataframe.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(dataframe.columns)
        
        text_color = QtGui.QColor(255, 255, 255)
        
        for i in range(dataframe.shape[0]):
            for j in range(dataframe.shape[1]):
                item_value = str(dataframe.iloc[i, j])
                item = QtWidgets.QTableWidgetItem(item_value)
                item.setForeground(QtGui.QBrush(text_color))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.tableWidget.setItem(i, j, item)
                
    def select_model(self):
        dialog = ModelSelectionDialog()
        
        # Get the main window's geometry
        main_window_geometry = self.SelectModel.parentWidget().geometry()
        main_window_position = self.SelectModel.parentWidget().parentWidget().frameGeometry().topLeft()

        # Calculate where to place the dialog
        dialog_x = main_window_position.x() + main_window_geometry.width()
        dialog_y = main_window_position.y()
        dialog.move(dialog_x, dialog_y)
        
        # Execute the dialog
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Store the selected model name
            self.selected_model = dialog.selected_model()
            if self.selected_model is not None:
                self.model_name_label = QtWidgets.QLabel(f"Selected Model: {self.selected_model}", self.widget)
                self.model_name_label.setGeometry(QtCore.QRect(720, 180, 160, 30))
                self.model_name_label.setStyleSheet("color: #ffffff;")
                self.model_name_label.show()
            else:
                self.model_name_label = QtWidgets.QLabel('', self.widget)
                self.model_name_label.setGeometry(QtCore.QRect(720, 180, 160, 30))
                self.model_name_label.setStyleSheet("color: #ffffff;")
                self.model_name_label.show()
    
    def select_analyse_target(self):
        if self.selected_file_path is None or not os.path.isfile(self.selected_file_path):
            self.show_warning_message("Please select a CSV file.")
            return

        try:
            df = pd.read_csv(self.selected_file_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"An error occurred while reading the file: {e}")
            return

        column_names = df.columns.tolist()
        dialog = SelectAnalyseTargetDialog(column_names, self.selected_file_path)

        # Set dialog position and modality
        self.set_dialog_position_and_modality(dialog)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_data_columns, selected_target = dialog.get_selected_columns()
            if selected_data_columns and selected_target:
                self.selected_data_columns = selected_data_columns
                self.selected_target = selected_target
                self.display_selected_target()  # Display the selected target
                self.display_selected_columns()  # Display the selected columns
                self.update_table_data()  # Update the table data with the selected columns and target
                self.handle_selected_data()  # Handle the selected data
        
    def set_dialog_position_and_modality(self, dialog):
        main_window_geometry = self.SelectModel.parentWidget().geometry()
        main_window_position = self.SelectModel.parentWidget().parentWidget().frameGeometry().topLeft()
        dialog_x = main_window_position.x() + main_window_geometry.width()
        dialog_y = main_window_position.y()
        dialog.move(dialog_x, dialog_y)
        dialog.setWindowModality(Qt.ApplicationModal)

    def display_selected_target(self):
        if hasattr(self, 'analyse_target_label'):
            self.analyse_target_label.deleteLater()
        self.analyse_target_label = QtWidgets.QLabel(f"Selected Target: {self.selected_target}", self.widget)
        self.analyse_target_label.setGeometry(QtCore.QRect(720, 210, 160, 30))  # Adjust the size to fit the content
        self.analyse_target_label.setStyleSheet("color: #ffffff;")
        self.analyse_target_label.show()
        
    def display_selected_columns(self):
        formatted_columns = "\n".join(self.selected_data_columns)
        if hasattr(self, 'analyse_columns_label'):
            self.analyse_columns_label.deleteLater()
        self.analyse_columns_label = QtWidgets.QLabel(f"Selected Data: {formatted_columns}", self.widget)
        self.analyse_columns_label.setGeometry(QtCore.QRect(720, 240, 160, 90))  # Adjusted height to accommodate multiple lines
        self.analyse_columns_label.setStyleSheet("color: #ffffff;")
        self.analyse_columns_label.show()
        
                
    def handle_selected_data(self):
        if hasattr(self, 'selected_file_path') and hasattr(self, 'selected_target'):
            # Here you can load the data from the CSV and process it using the selected target
            try:
                df = pd.read_csv(self.selected_file_path)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"An error occurred while reading the file: {e}")
    
    def update_table_data(self):
        if hasattr(self, 'selected_data_columns') and hasattr(self, 'selected_target'):
            try:
                df = pd.read_csv(self.selected_file_path)
                # Filter the dataframe to include only the selected columns and target
                selected_columns = self.selected_data_columns + [self.selected_target]
                filtered_df = df[selected_columns]
                # Now load the filtered data into the table
                self.load_data_into_table(filtered_df)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"An error occurred while updating the table: {e}")
        
    def show_warning_message(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText(message)
        msg.setIcon(QMessageBox.Warning)
        # Assuming `icon_path` is defined somewhere in your class
        msg.setStyleSheet("background-color: #301860; color: #ffffff;")
        msg.setWindowIcon(QtGui.QIcon(icon_path))
        msg.exec_()  # Display the message box
    
    def start_analyse(self):
        # Check if all necessary data is selected
        if not hasattr(self, 'selected_file_path') or self.selected_file_path is None:
            self.show_warning_message("Please select a CSV file.")
            return
        if not hasattr(self, 'selected_target') or self.selected_target is None:
            self.show_warning_message("Please select a target.")
            return
        if not hasattr(self, 'selected_data_columns') or not self.selected_data_columns:
            self.show_warning_message("Please select data columns.")
            return
        if not hasattr(self, 'selected_model') or self.selected_model is None:
            self.show_warning_message("Please select a model.")
            return

        # Start the worker task
        self.handler_worker.setup_data(
            self.selected_file_path,
            self.selected_target,
            self.selected_data_columns,
            self.selected_model
        )
        # Show the loading screen/message
        self.show_loading_dialog("Processing... Please wait.")
        # Start the worker's thread
        self.handler_worker_thread.start()

    def show_loading_dialog(self, message):
        self.loading_dialog = QtWidgets.QProgressDialog(self.Form, QtCore.Qt.WindowTitleHint)
        self.loading_dialog.setLabelText(message)
        self.loading_dialog.setStyleSheet("color: #ffffff;")
        self.loading_dialog.setCancelButton(None)
        self.loading_dialog.setRange(0, 0)
        self.loading_dialog.setWindowTitle("Loading")
        self.loading_dialog.setModal(True)
        self.loading_dialog.show()
        
        # Ensure GUI updates by processing events
        QtCore.QCoreApplication.processEvents()

    def on_handling_finished(self):
        # Close the loading dialog
        self.loading_dialog.close()

        # Quit the thread
        self.handler_worker_thread.quit()
        self.handler_worker_thread.wait()

        # The show_success_message method is called after the canvas is shown
        # Schedule the message box to show after the event loop has returned
        QtCore.QTimer.singleShot(0, self.show_success_message)
  
    def show_success_message(self):
        # Show message box that handling is finished
        msg = QMessageBox()
        msg.setWindowTitle("Success")
        msg.setText("Analyse complete.")
        msg.setIcon(QMessageBox.Information)
        msg.setStyleSheet("background-color: #301860; color: #ffffff;")
        msg.setWindowIcon(QtGui.QIcon(icon_path))
        msg.setWindowModality(QtCore.Qt.NonModal)  # Allow interaction with other windows
        msg.exec_()

    def on_data_ready(self, fig):
        canvas = FigureCanvas(fig)
        
        # Create a new QDialog to host the canvas
        self.figure_dialog = QtWidgets.QDialog()
        self.figure_dialog.setWindowTitle("Figure")  # Give it a title
        
        # Set the window icon
        self.figure_dialog.setWindowIcon(QtGui.QIcon(icon_path))  # Assuming icon_path is defined
        
        # Set up the layout and add the canvas
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(canvas)
        self.figure_dialog.setLayout(layout)
        
        # Apply the desired background and text colors using stylesheets
        self.figure_dialog.setStyleSheet("QDialog { background-color: #301860; color: #ffffff; }")
        
        # Calculate where to place the dialog
        main_window_position = self.Form.pos()  # Get the position of the main window
        main_window_geometry = self.Form.geometry()  # Get the geometry of the main window
        dialog_x = main_window_position.x() + main_window_geometry.width()
        dialog_y = main_window_position.y()
        self.figure_dialog.move(dialog_x, dialog_y)  # Move the dialog to the calculated position
        
        # Show the dialog
        self.figure_dialog.show()
        
        # Store the canvas and dialog so they aren't garbage collected
        self.canvas = canvas
        self.canvas_dialog = self.figure_dialog

if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    
    # Set fixed size to the current size
    Form.setFixedSize(Form.size())

    # Set the window flags here
    Form.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
    
    # Set the icon
    icon_path = os.path.join(os.path.dirname(__file__), '../media/icon.png')
    if not os.path.exists(icon_path):
        print(f"Icon file not found at path: {icon_path}")
    else:
        Form.setWindowIcon(QtGui.QIcon(icon_path))
    
    Form.show()
    sys.exit(app.exec_())