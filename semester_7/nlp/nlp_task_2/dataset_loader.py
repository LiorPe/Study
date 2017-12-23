from datetime import datetime

twits_file_path = 'Data/tweets.tsv'
trump_term_beginning_date = datetime(year=2017, month=1, day=20)
trump_switched_to_iphone_date = datetime(year=2017, month=4, day=1)

comments_file_path = 'Data/proc_17_108_unique_comments_text_dupe_count.csv'

class Twit:
    def __init__(self, id, user_handle, text, time_stamp, device):
        self.id = id
        self.user_handle = user_handle
        self.device = device
        self.text = text
        self.time_stamp = datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')
        self.is_in_trump_term = self.time_stamp >= trump_term_beginning_date
        self.is_after_trump_switched_to_iphone = self.time_stamp >= trump_switched_to_iphone_date

        self.apply_label()

    def apply_label(self):
        if self.user_handle == 'PressSec' or self.user_handle == 'POTUS' :
            self.label = 0
        elif self.user_handle == 'realDonaldTrump' and self.device == 'iphone' and not self.is_after_trump_switched_to_iphone:
            self.label = 0
        elif self.user_handle == 'realDonaldTrump' and self.device == 'android' and not self.is_after_trump_switched_to_iphone:
            self.label = 1
        else:
            self.label = None


def read_twits_file():
    all_twits = []
    with open(twits_file_path) as file:
        for line in file:
            split_line = line.strip('\n').split('\t')
            id = split_line[0]
            user_handle = split_line[1]
            text = split_line[2]
            time_stamp = split_line[3]
            device = split_line[4]
            cur_twitt = Twit(id, user_handle, text, time_stamp, device)
            all_twits.append(cur_twitt)
    return all_twits


def get_twits_data():
    all_twits = read_twits_file()
    return [twit.text for twit in all_twits]


def new_record(split_line):
    try:
        int(split_line[0])
        return True
    except:
        return False


def add_line_to_current_sentence(split_line):
    try:
        float(split_line[-1])
        return ','.join(split_line[:-1])
    except:
        return ','.join(split_line)



def get_comments_data():
    all_sentences = []
    first_line = True
    current_sentence = ''
    with open (comments_file_path) as file:
        for line in file:
            if first_line:
                first_line = False
                continue
            split_line = line.split(',')
            if new_record(split_line):
                if current_sentence != '':
                    all_sentences.append(current_sentence)
                current_sentence = ''
                current_sentence+= add_line_to_current_sentence(split_line)
    return all_sentences


# def main():
#     get_comments_data()
#
# if __name__ == main():
#     main()
