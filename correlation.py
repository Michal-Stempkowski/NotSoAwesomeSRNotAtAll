from functools import reduce
import os
from audiolazy.lazy_lpc import lpc
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
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)

    @staticmethod
    def from_list(ls):
        return Point(ls[0], ls[1], ls[2])


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
        probes = list(curve(range(size)))

        plot_shit(probes)
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
        lpc_curve = lpc(Signal.convolution(sample), num_of_formants+2)
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
            best = (other, Point.from_list(take).distance(point), name)
        else:
            curr = (other, Point.from_list(take).distance(point), name)
            best = best if best[1] <= curr[1] else curr

    return best


def plot_shit(sh):
    plot.plot(sh)
    plot.ylabel('Literka')
    plot.show()

if __name__ == '__main__':
    fonem_dir = 'fonems'
    num_of_formants = 3

    fonem_provider = get_fonem_provider(fonem_dir)
    schemas = load_characteristics(fonem_provider, num_of_formants)

    takes = get_fonem_provider('fonems2')

    tested = 0
    well_classified = 0

    t1 = list(get_fonem_provider('fonems'))
    t2 = list(get_fonem_provider('fonems2'))

    test_schemas = [(fonem_name, Signal.generate_characteristics_of_sample_with_lpc(fonem_sample, num_of_formants))
            for (fonem_name, fonem_sample) in t1]

    test_schemas2 = [(fonem_name, Signal.generate_characteristics_of_sample_with_lpc(fonem_sample, num_of_formants))
            for (fonem_name, fonem_sample) in t2]



    lpced = lpc(t1[0][1], num_of_formants)
    lpced.plot().show()

    print(Signal.generate_characteristics_of_sample_with_lpc(t1[0][1], num_of_formants))

    input()


    # print(test_schemas[0][1])
    # print(test_schemas[0][1])
    # print(test_schemas2[0][1])

    # for i in range(len(test_schemas)):
    #     print(test_schemas[i][0])
    #     print(test_schemas[i][1])
    #     print(test_schemas2[i][1])
        # list(test_schemas[0][1](range(1000)))
        # lpc(t1[0][1], num_of_formants+2).plot().show()
        # input()

    # lpc(t1[0][1], num_of_formants + 2).plot().show()
    # lpc(t2[0][1], num_of_formants + 2).plot().show()
    # lpc(t1[1][1], num_of_formants + 2).plot().show()
    # lpc(t2[1][1], num_of_formants + 2).plot().show()
    # input()


    # for a in range(len(test_schemas)):
    #     print(test_schemas[a][0])
    #     for i in range(len(test_schemas)):
    #         test_schemas[a][1].plot().show()
    #         test_schemas[i][1].plot().show()
    #         print(test_schemas[a][1].diff() - test_schemas2[i][1].diff())
    #         break
    #
    #     print()
        # input()

    # for (name, take) in takes:
    #     recognized = recognize_character_lpc(take, schemas, num_of_formants+2)
    #
    #     result = recognized == name
    #     tested += 1
    #     well_classified += 1 if result else 0
    #     print(name, '-->', recognized, 'SUCCESS!!' if result else '')

    # for (name, take) in takes:
    #     recognized = recognize_character(take, schemas)
    #
    #     result = recognized == name
    #     tested += 1
    #     well_classified += 1 if result else 0
    #     print(name, '-->', recognized, 'SUCCESS!!' if result else '')
    #
    #     plot.plot(take)
    #     plot.show()
    #
    #     lpc(take, order=4).plot().show()

    print('Recognition: ', well_classified, '/', tested)

    print('Hello World!')