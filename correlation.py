from functools import reduce
import os
from numpy.fft import rfft

import matplotlib.pyplot as plot
import math


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


def calculate_delta(rate):
    return rate // 52


def calculate_error(signal_characteristics, sample_characteristics):
    return reduce(lambda x, y: x + y, (math.fabs(signal - sample) for
                                       (signal, sample) in zip(signal_characteristics, sample_characteristics))) \
        / len(signal_characteristics)


count = 0


def load_characteristics(fonem_provider, number_of_formants):
    return [(fonem_name, Signal.generate_characteristics_of_sample(fonem_sample, number_of_formants))
            for (fonem_name, fonem_sample) in fonem_provider]


def recognize_character(signal, fonem_schemas):
    number_of_formants = len(fonem_schemas[0][1])
    signal_schema = Signal.generate_characteristics_of_sample(signal, number_of_formants)
    error_measure = [(name, calculate_error(signal_schema, schema)) for (name, schema) in fonem_schemas]
    print(error_measure)
    return reduce(lambda best, curr: best if best[1] < curr[1] else curr, error_measure)[0]


def read_fonem_file(filename, fonem_dir):
    with open(fonem_dir + os.sep + filename) as file:
        return filename.split('.')[0], eval(file.readlines()[0])


def get_fonem_provider(fonem_dir):
    return (read_fonem_file(filename, fonem_dir)
            for filename in os.listdir(fonem_dir) if filename.find('.fonem') >= 0)


if __name__ == '__main__':
    fonem_dir = 'fonems'
    num_of_formants = 4

    fonem_provider = get_fonem_provider(fonem_dir)
    schemas = load_characteristics(fonem_provider, num_of_formants)

    takes = get_fonem_provider('fonems2')

    tested = 0
    well_classified = 0

    for (name, take) in takes:
        recognized = recognize_character(take, schemas)

        result = recognized == name
        tested += 1
        well_classified += 1 if result else 0
        print(name, '-->', recognized, 'SUCCESS!!' if result else '')

    print('Recognition: ', well_classified, '/', tested)

    print('Hello World!')