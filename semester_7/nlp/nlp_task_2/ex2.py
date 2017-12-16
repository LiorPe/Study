import sys
print(sys.version)

twits_file_path ='Data/tweets.tsv'
class Twit:
    def __init__(self,id, user_handle,text, time_stamp, device  ):
        self.id = id
        self.user_handle =user_handle
        self.text =text
        self.time_stamp =time_stamp
        self.device =device
        self.label = None

def read_file(tweets_file_path):
    all_twits = []
    with open(tweets_file_path) as file:
        for line in file:
            split_line = line.split('\t')
            id = split_line[0]
            user_handle = split_line[1]
            text = split_line[2]
            time_stamp = split_line[3]
            device = split_line[4]
            cur_twitt = Twit(id,user_handle,text,time_stamp, device)
            all_twits.append(cur_twitt)

