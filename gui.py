import sys
from functools import partial
import threading
from array import array
from Queue import Queue, Full
import matplotlib.pyplot as plot
import pyaudio
import math
from time import sleep
from PyQt4 import QtGui, QtCore
import correlation

CHUNK_SIZE = 1024
MIN_VOLUME = 500
count = 0
BUF_MAX_SIZE = CHUNK_SIZE * 15
fonem_provider = None
schemas = None

characters = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
              'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
              'z', 'x', 'c', 'v', 'b', 'n', 'm',
              "space", ]

buttons = {}


class AThread(QtCore.QThread):
    def __init__(self):
        QtCore.QThread.__init__(self)
        self.signal = QtCore.SIGNAL("signal")

    def run(self):
        stopped = threading.Event()
        q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK_SIZE)))

        listen_t = threading.Thread(target=self.listen, args=(stopped, q))
        listen_t.start()
        record_t = threading.Thread(target=self.record, args=(stopped, q))
        record_t.start()

        try:
            while True:
                listen_t.join(0.1)
                record_t.join(0.1)
        except KeyboardInterrupt:
            stopped.set()

        listen_t.join()
        record_t.join()

    def record(self, stopped, q):
        previous = False
        speech_arr = []
        while True:
            if stopped.wait(timeout=0):
                break
            chunk = q.get()
            vol = max(chunk)
            if vol >= MIN_VOLUME:
                speech_arr += chunk
                previous = True
            else:
                if previous:
                    self.print_speech(speech_arr)
                    speech_arr = []
                    sleep(2)
                previous = False

    def print_speech(self, speech_arr):
        global keyboard
        print(speech_arr)
        self.emit(self.signal, correlation.recognize_character(speech_arr, schemas))
        #plot.plot(speech_arr)
        #plot.ylabel('Literka')
        #plot.show()

    def listen(self, stopped, q):
        stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
        )

        while True:
            if stopped.wait(timeout=0):
                break
            try:
                q.put(array('h', stream.read(CHUNK_SIZE)))
            except Full:
                pass  # discard


class Keyboard(QtGui.QWidget):
    previous_button = "a"

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.resize(1090, 250)
        self.move(300, 300)
        self.setWindowTitle('Speech Keyboard')

        self.editBox = QtGui.QLineEdit(self)
        self.editBox.setPlaceholderText('Written text')
        self.editBox.setMinimumWidth(1050)
        self.editBox.move(20, 10)

        for x in range(len(characters) - 17):
            buttons[x] = QtGui.QPushButton(characters[x], self)
            buttons[x].move(20 + 105 * x, 60)

        for x in range(10, len(characters) - 8):
            buttons[x] = QtGui.QPushButton(characters[x], self)
            buttons[x].move(70 + 105 * (x - 10), 100)

        for x in range(19, len(characters) - 1):
            buttons[x] = QtGui.QPushButton(characters[x], self)
            buttons[x].move(70 + 105 * (x - 18), 140)

        buttons[26] = QtGui.QPushButton(characters[26], self)
        buttons[26].setFixedWidth(500)
        buttons[26].move(290, 180)

        for x in range(len(characters)):
            buttons[x].clicked.connect(partial(self.push_button, string=characters[x]))

        self.threadPool = []

    def push_button(self, string):
        buttons[button_mapping(self.previous_button)].setStyleSheet("background-color: light gray")
        buttons[button_mapping(string)].setStyleSheet("background-color: green")
        text = self.get_edit()
        if string == "space":
            self.set_edit(text + " ")
        else:
            self.set_edit(text + string)
        self.previous_button = string

    def set_edit(self, text):
        self.editBox.setText(text)

    def get_edit(self):
        return self.editBox.text()

    def start_recording_threads(self):
        self.thread = AThread()
        self.connect(self.thread, self.thread.signal, self.push_button)
        self.thread.start()


def button_mapping(string):
    return {
        'q': 0, 'w': 1, 'e': 2, 'r': 3, 't': 4, 'y': 5, 'u': 6, 'i': 7, 'o': 8, 'p': 9, 'a': 10,
        's': 11, 'd': 12, 'f': 13, 'g': 14, 'h': 15, 'j': 16, 'k': 17, 'l': 18, 'z': 19, 'x': 20,
        'c': 21, 'v': 22, 'b': 23, 'n': 24, 'm': 25, "space": 26
    }.get(string, -1)


def gui_main():
    global schemas
    global fonem_provider

    app = QtGui.QApplication(sys.argv)
    keyboard = Keyboard()
    keyboard.show()

    # keyboard.push_button("a")
    #keyboard.push_button("h")

    fonem_dir = 'fonems'
    num_of_formants = 4

    fonem_provider = correlation.get_fonem_provider(fonem_dir)
    schemas = correlation.load_characteristics(fonem_provider, num_of_formants)

    keyboard.start_recording_threads()

    sys.exit(app.exec_())


if __name__ == '__main__':
    gui_main()
