"""Handwriting UI; inference is performed by the GCC-built C executable."""
import json
from pathlib import Path
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
GRID = 28
CELL = 10
SIZE = GRID * CELL


def prepare_pixels(pixels):
    """Pass the 28x28 grid unchanged to the C model."""
    return bytes(pixels) if any(pixels) else None


class DigitApp:
    def __init__(self, window):
        self.window = window
        self.pixels = bytearray(GRID * GRID)
        self.last = None
        self.busy = False
        self.results = queue.Queue()
        window.title('手写数字识别 · C CNN')
        window.resizable(False, False)
        frame = ttk.Frame(window, padding=24)
        frame.grid()
        ttk.Label(frame, text='手写数字识别', font=('', 22, 'bold')).grid(row=0, column=0, sticky='w')
        ttk.Label(frame, text='28×28 像素格 · 左键书写 / 右键擦除 · 原样识别', foreground='#555555').grid(
            row=1, column=0, columnspan=2, sticky='w', pady=(6, 18))
        self.canvas = tk.Canvas(frame, width=SIZE, height=SIZE, bg='black',
                                highlightthickness=1, highlightbackground='#777777', cursor='crosshair')
        self.canvas.grid(row=2, column=0, sticky='n')
        self.create_grid()
        self.canvas.bind('<Button-1>', self.begin)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', lambda event: setattr(self, 'last', None))
        self.canvas.bind('<Button-3>', lambda event: self.begin(event, erase=True))
        self.canvas.bind('<B3-Motion>', lambda event: self.draw(event, erase=True))
        self.canvas.bind('<ButtonRelease-3>', lambda event: setattr(self, 'last', None))
        panel = ttk.Frame(frame, padding=(24, 0, 0, 0))
        panel.grid(row=2, column=1, rowspan=2, sticky='n')
        self.answer = tk.StringVar(value='—')
        ttk.Label(panel, text='识别结果').grid(row=0, column=0)
        ttk.Label(panel, textvariable=self.answer, font=('', 48, 'bold')).grid(row=1, column=0)
        ttk.Label(panel, text='模型输入 · 28 × 28').grid(row=2, column=0, pady=(6, 6))
        self.preview = tk.Label(panel, bg='black', width=112, height=112)
        self.preview.grid(row=3, column=0)
        self.show_preview(bytes(784))
        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, sticky='ew', pady=12)
        self.recognize_button = ttk.Button(buttons, text='识别', command=self.recognize)
        self.recognize_button.pack(side='left', fill='x', expand=True)
        self.clear_button = ttk.Button(buttons, text='清空重写', command=self.clear)
        self.clear_button.pack(side='left', fill='x', expand=True, padx=(10, 0))
        self.status = tk.StringVar(value='准备好了，请在黑色区域书写。')
        ttk.Label(frame, textvariable=self.status).grid(row=4, column=0, columnspan=2, sticky='w', pady=8)
        model_frame = ttk.Frame(frame)
        model_frame.grid(row=5, column=0, columnspan=2, sticky='ew', pady=8)
        default = ROOT / 'CNN/CNN/my_model.cnn'
        if not default.exists():
            default = ROOT / 'CNN/CNN/minst.cnn'
        self.model = tk.StringVar(value=str(default))
        ttk.Label(model_frame, text='模型：').pack(side='left')
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model, width=44)
        self.model_entry.pack(side='left', fill='x', expand=True)
        self.choose_button = ttk.Button(model_frame, text='选择…', command=self.choose_model)
        self.choose_button.pack(side='left', padx=(8, 0))
        ttk.Label(frame, text='各类别平方距离（越小越匹配，不是概率）').grid(
            row=6, column=0, columnspan=2, sticky='w', pady=(12, 4))
        self.chart = tk.Canvas(frame, width=470, height=122, bg='#f4f5f7', highlightthickness=0)
        self.chart.grid(row=7, column=0, columnspan=2)
        self.show_scores([0] * 10)
        window.bind('<Return>', lambda event: self.recognize())
        window.after(50, self.poll_result)

    def choose_model(self):
        name = filedialog.askopenfilename(initialdir=ROOT / 'CNN/CNN',
                                         filetypes=[('CNN 模型', '*.cnn')])
        if name:
            self.model.set(name)
            self.answer.set('—')
            self.show_scores([0] * 10)
            self.status.set('已切换模型，请重新识别。')

    def create_grid(self):
        self.cells = []
        for y in range(GRID):
            for x in range(GRID):
                self.cells.append(self.canvas.create_rectangle(
                    x * CELL, y * CELL, (x + 1) * CELL, (y + 1) * CELL,
                    fill='black', outline='#353535'))
        # 田字格中线仅作定位参考，不写入像素数据。
        middle = SIZE // 2
        self.canvas.create_line(middle, 0, middle, SIZE, fill='#a56838', width=2)
        self.canvas.create_line(0, middle, SIZE, middle, fill='#a56838', width=2)

    def begin(self, event, erase=False):
        if self.busy:
            return
        self.last = None
        self.answer.set('—')
        self.show_scores([0] * 10)
        self.status.set('每格对应一个像素，写好后点击「识别」。')
        self.draw(event, erase)

    def draw(self, event, erase=False):
        if self.busy:
            return
        if not (0 <= event.x < SIZE and 0 <= event.y < SIZE):
            self.last = None
            return
        x, y = event.x // CELL, event.y // CELL
        start_x, start_y = self.last or (x, y)
        distance = max(1, abs(x - start_x), abs(y - start_y))
        for step in range(distance + 1):
            px = round(start_x + (x - start_x) * step / distance)
            py = round(start_y + (y - start_y) * step / distance)
            index = py * GRID + px
            self.pixels[index] = 0 if erase else 255
            self.canvas.itemconfigure(self.cells[index], fill='black' if erase else 'white')
        self.last = x, y
        self.show_preview(self.pixels)

    def clear(self):
        if self.busy:
            return
        self.pixels = bytearray(GRID * GRID)
        self.last = None
        for cell in self.cells:
            self.canvas.itemconfigure(cell, fill='black')
        self.answer.set('—')
        self.show_scores([0] * 10)
        self.show_preview(bytes(784))
        self.status.set('画布已清空，请写一个数字。')

    def show_preview(self, pixels):
        photo = tk.PhotoImage(width=28, height=28)
        rows = ['{' + ' '.join(f'#{v:02x}{v:02x}{v:02x}' for v in pixels[y*28:(y+1)*28]) + '}'
                for y in range(28)]
        photo.put(' '.join(rows))
        self.preview_photo = photo.zoom(4)
        self.preview.configure(image=self.preview_photo)

    def show_scores(self, scores, best=None):
        self.chart.delete('all')
        self.chart.create_line(0, 92, 470, 92, fill='#cccccc')
        # 距离不再受tanh范围限制；按本次最大距离缩放柱高。
        maximum = max(max(scores, default=0), 1)
        for i, score in enumerate(scores):
            x = 10 + i * 46
            color = '#167d70' if i == best else '#9eaebb'
            self.chart.create_rectangle(x, 92 - score / maximum * 60, x + 30, 92,
                                        fill=color, outline='')
            self.chart.create_text(x + 15, 13, text=f'{score:.1f}', font=('', 9))
            self.chart.create_text(x + 15, 104, text=str(i), font=('', 11, 'bold'))

    def set_busy(self, busy):
        self.busy = busy
        for widget in (self.recognize_button, self.clear_button, self.choose_button, self.model_entry):
            widget.configure(state='disabled' if busy else 'normal')

    def recognize(self):
        if self.busy:
            return
        pixels = prepare_pixels(self.pixels)
        if pixels is None:
            self.status.set('请先写一个数字，空白画布不能识别。')
            return
        self.show_preview(pixels)
        model = Path(self.model.get()).expanduser()
        if not model.is_file():
            messagebox.showerror('模型不存在', f'找不到模型文件：\n{model}')
            return
        self.set_busy(True)
        self.status.set('正在调用 C 模型识别…')
        self.answer.set('—')
        self.show_scores([0] * 10)
        threading.Thread(target=self.run_model, args=(pixels, model), daemon=True).start()

    def run_model(self, pixels, model):
        try:
            result = subprocess.run([str(ROOT / 'build/cnn'), 'predict', str(model)],
                                    input=pixels, capture_output=True, timeout=15, check=True)
            self.results.put((json.loads(result.stdout), None))
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            detail = error.stderr.decode(errors='replace') if isinstance(error, subprocess.CalledProcessError) else str(error)
            self.results.put((None, detail))

    def poll_result(self):
        try:
            result, error = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.set_busy(False)
            if error:
                self.status.set('识别失败，请检查模型或运行 make 重新编译。')
                messagebox.showerror('识别失败', error)
            else:
                self.answer.set(str(result['digit']))
                self.show_scores(result['scores'], result['digit'])
                self.status.set('识别完成。可继续补画，或清空后写下一个数字。')
        self.window.after(50, self.poll_result)


if __name__ == '__main__':
    window = tk.Tk()
    DigitApp(window)
    window.mainloop()
