from functools import reduce
import os
from audiolazy.lazy_filters import z
from audiolazy.lazy_lpc import lpc
from audiolazy.lazy_math import dB20
from audiolazy.lazy_synth import line
from numpy.fft import rfft

import matplotlib.pyplot as plot
import math
from numpy.ma import abs


class Point(object):
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    @staticmethod
    def from_list(ls):
        # ls = list(reversed(ls))
        index = 0
        return Point(ls[0 + index], ls[1+ index], ls[2+ index]) if len(ls) > 2+ index else Point(ls[0+ index], ls[1+ index], ls[1+ index])


    def __str__(self):
        return '{0:.2f}x{1:.2f}x{2:.2f}'.format(self.x, self.y, self.z)

    __repr__ = __str__


class Signal(object):

    @staticmethod
    def pad_with_zeroes(signal, padding):
        return signal + [0 for _ in range(padding - len(signal))]

    @staticmethod
    def convolution(signal):
        return [math.fabs(x) for x in rfft(Signal.pad_with_zeroes(signal, len(signal) * 2))]

    @staticmethod
    def normalize(signal):
        max_value = max(signal)
        return [x / max_value for x in signal]

    @staticmethod
    def find_formant(signal, denied):
        return reduce(
            lambda cur_max, i:
            cur_max if denied[i] or cur_max[0] > signal[i]
            else (signal[i], i), range(len(signal)), (-1, None)
        )[1]

    @staticmethod
    def find_k_formants(signal, k, delta):
        denied = [False for _ in range(len(signal))]

        for _ in range(k):
            new_formant = Signal.find_formant(signal, denied)

            for i in range(new_formant - delta, new_formant + delta):
                denied[i] = True

            yield new_formant

    @staticmethod
    def generate_characteristics_of_sample(sample, number_of_formants):
        conv = Signal.convolution(sample)
        delta = calculate_delta(len(conv))
        return sorted(Signal.find_k_formants(conv, number_of_formants, delta))

    @staticmethod
    def find_peaks_on_curve(curve, number_of_peaks):
        peaks = list()

        greater = lambda x, y: x >= y
        equal = lambda x, y: x == y
        lower = lambda x, y: x <= y

        update_state = lambda x, y, previous_state: \
            previous_state if previous_state and previous_state(x, y) \
            else (greater if greater(x, y) else lower)

        update_peaks = lambda old_state, new_state: \
            old_state != new_state and old_state != equal and new_state != equal

        previous_state = None

        size = 1000
        # curve.plot().show()
        # input()
        probes = extract_lpc_numerals(curve, size)
        # plot_shit(probes)
        # input()
        # probes = list(curve(range(size)))

        # tab = range (1000)
        # filtr =curve
        # result = curve(tab)
        # plot.plot(list(result))
        # plot.show()


        for i in range(size - 1):
            x = probes[i]
            y = probes[i + 1]
            next_state = update_state(x, y, previous_state)
            if update_peaks(previous_state, next_state):
                peaks.append(x)

            previous_state = next_state

        return peaks[:]


    @staticmethod
    def generate_characteristics_of_sample_with_lpc(sample, number_of_formants):
        lpc_curve = lpc(
            Signal.convolution(sample),
            number_of_formants)
        return Signal.find_peaks_on_curve(lpc_curve, number_of_formants)


def calculate_delta(rate):
    return rate // 52


def calculate_error(signal_characteristics, sample_characteristics):
    return reduce(lambda x, y: x + y, (math.fabs(signal - sample) for
                                       (signal, sample) in zip(signal_characteristics, sample_characteristics))) \
        / len(signal_characteristics)


count = 0


def load_characteristics(fonem_provider, number_of_formants):
    return [(fonem_name, Signal.generate_characteristics_of_sample_with_lpc(fonem_sample, number_of_formants))
            for (fonem_name, fonem_sample) in fonem_provider]


def recognize_character(signal, fonem_schemas):
    number_of_formants = len(fonem_schemas[0][1])
    signal_schema = Signal.generate_characteristics_of_sample_with_lpc(signal, number_of_formants)
    error_measure = [(name, calculate_error(signal_schema, schema)) for (name, schema) in fonem_schemas]
    print(error_measure)
    return reduce(lambda best, curr: best if best[1] < curr[1] else curr, error_measure)[0]


def read_fonem_file(filename, fonem_dir):
    with open(fonem_dir + os.sep + filename) as file:
        return filename.split('.')[0], eval(file.readlines()[0])


def get_fonem_provider(fonem_dir):
    return (read_fonem_file(filename, fonem_dir)
            for filename in os.listdir(fonem_dir) if filename.find('.fonem') >= 0)


def recognize_character_lpc(take, schemas, num_of_formants):
    characteristics = Signal.generate_characteristics_of_sample_with_lpc(take, num_of_formants)

    point = Point.from_list(characteristics)
    best = None
    for (name, other) in schemas:
        if not best:
            best = (other, Point.from_list(other).distance(point), name)
        else:
            curr = (other, Point.from_list(other).distance(point), name)
            best = best if best[1] <= curr[1] else curr

    return best


def plot_shit(sh):
    plot.plot(sh)
    plot.ylabel('Literka')
    plot.show()

def extract_lpc_numerals(lpc_result, samples):
    min_freq=0.
    max_freq=3.141592653589793;
    freq_scale="linear"
    mag_scale="dB"
    fscale = freq_scale.lower()
    mscale = mag_scale.lower()
    mscale = "dB"

    Hz = 3.141592653589793 / 12.

    freqs = list(line(samples, min_freq, max_freq, finish=True))
    freqs_label = list(line(samples, min_freq / Hz, max_freq / Hz, finish=True))
    data = lpc_result.freq_response(freqs)
    mag = { "dB": dB20 }[mscale]

    # print("extract numerals")
    return (mag(data))

if __name__ == '__main__':
    fonem_dir = 'fonems'
    num_of_formants = 4

    fonem_provider = get_fonem_provider(fonem_dir)
    schemas = load_characteristics(fonem_provider, num_of_formants)


    takes = get_fonem_provider('fonems2')

    tested = 0
    well_classified = 0

    t1 = list(get_fonem_provider('fonems'))#[:2]
    t2 = list(get_fonem_provider('fonems2'))#[:2]

    test_schemas = [(fonem_name, Signal.generate_characteristics_of_sample_with_lpc(fonem_sample, num_of_formants))
            for (fonem_name, fonem_sample) in t1][:1]

    test_schemas2 = [(fonem_name, Signal.generate_characteristics_of_sample_with_lpc(fonem_sample, num_of_formants))
            for (fonem_name, fonem_sample) in t2]

    i = 0
    for (name, take) in takes:
        recognized = recognize_character_lpc(take, schemas, num_of_formants)

        result = recognized[2] == name
        tested += 1
        well_classified += 1 if result else 0
        print(name,
              # Point.from_list(schemas[i][1]),
              '-->',
              recognized[2],
              'SUCCESS!!' if result else '')

        i += 1


    print('Recognition: ', well_classified, '/', tested)

    print('Hello World!')