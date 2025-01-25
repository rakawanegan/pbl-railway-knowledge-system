import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd

# Create the main application window
root = tk.Tk()
root.title("Improved CSV Viewer with Transposed Rows")
root.geometry("1000x600")

# Initialize global variables
df = None
current_row = 0

# Function to load a CSV file
def load_csv():
    global df, current_row
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            current_row = 0
            display_row()
        except Exception as e:
            tk.messagebox.showerror("Error", f"Unable to read the CSV file. {e}")

# Function to display the current row transposed in a Text widget
def display_row():
    global current_row
    if df is not None:
        text_widget.delete(1.0, tk.END)  # Clear previous content
        row_data = df.iloc[current_row]
        for field, value in row_data.items():
            text_widget.insert(tk.END, f"{field}:\n{value}\n\n")  # Add data with spacing

# Function to go to the next row
def next_row():
    global current_row
    if df is not None:
        if current_row < len(df) - 1:
            current_row += 1
            display_row()
        else:
            messagebox.showinfo("End of Rows", "You have reached the last row. Returning to the first row.")
            current_row = 0
            display_row()

# Function to go to the previous row
def previous_row():
    global current_row
    if df is not None:
        if current_row > 0:
            current_row -= 1
            display_row()
        else:
            messagebox.showinfo("Start of Rows", "You are at the first row. Cannot move back.")

# Function to quit the application
def quit_app():
    root.destroy()

# Create a menu bar
menu = tk.Menu(root)
root.config(menu=menu)

file_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open CSV", command=load_csv)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=quit_app)

# Create a frame for the text widget and scrollbar
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

text_widget = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, font=("Helvetica", 12))
text_widget.pack(fill=tk.BOTH, expand=True)

scrollbar.config(command=text_widget.yview)

# Add navigation buttons
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, pady=10)

prev_button = tk.Button(button_frame, text="Previous Row", command=previous_row, height=2, width=15)
prev_button.pack(side=tk.LEFT, padx=20)

next_button = tk.Button(button_frame, text="Next Row", command=next_row, height=2, width=15)
next_button.pack(side=tk.RIGHT, padx=20)

# Start the Tkinter event loop
root.mainloop()
