#!/usr/bin/python3
import os
import pwd
import subprocess as sp
import shlex
import time
from time import gmtime, strftime
import re
import mmap

def print_red(text):
    print("\033[91m{}\033[00m".format(text))

def execute_command(cmd, stdout=None, stderr=None):
    p = sp.Popen(shlex.split(cmd), stdout=stdout, stderr=stderr)
    out, err = p.communicate()
    retval = p.wait()

    return [out, err, retval]

class Single_Test:
    def __init__(self, hw, student_id):
        self.prev_wd = os.getcwd()
        self.pp_wd = "/nfs/.grade/{}".format(hw)
        self.student_id = student_id
        self.grade_wd = "{}/{}".format(self.pp_wd, self.student_id)
        self.src_tmpl = "src-grade"
        self.zip_name = "{}_{}.zip".format(hw, student_id)
        self.url_txt = "url.txt"
        self.score_dir = "{}/score".format(self.pp_wd)

        self.filter_correct = [True for i in range(3)]
        self.filter_correct_score = [25 for i in range(3)]
        self.filter_speedup = []
        self.filter_times = [0 for i in range(5)]
        self.filter_time = 0
        self.filter_perf = 0.0

    def checkZIP(self):
        if not os.path.isfile(self.zip_name):
            print_red("--------------------------------------")
            print_red("{} not exist!".format(self.zip_name))
            print_red("ID=\'{}\', ZIP=\'{}\'".format(self.student_id, self.zip_name))
            print_red("--------------------------------------")
            os.chdir(self.prev_wd)
            exit(0)

    def checkUrl(self):
        os.chdir("{}/{}".format(self.grade_wd, self.src_tmpl))

        if not os.path.isfile(self.url_txt):
            print_red("ID=\'{}\' you should provide url.txt.".format(self.student_id))
            return None

        with open(self.url_txt) as f:
            url = f.readline().strip("\n\r")

            if url.find("hackmd.io") == -1:
                print_red("ID=\'{}\' you should provide hackmd url in url.txt.".format(self.student_id))
                return None

            try:
                cmd = "curl {}".format(url)
                out, err, _ = execute_command(cmd, sp.PIPE, sp.PIPE)

                if out.decode('utf-8').find("<title>404 Not Found - HackMD</title>") != -1:
                    print_red("ID=\'{}\' the hackmd url you provide is not correct.".format(self.student_id))
                else:
                    print("ID=\'{}\' check url.txt  OK.".format(self.student_id))
            except:
                print_red("--------------------------------------")
                print_red("Curl url error!")
                print_red("ID=\'{}\', URL_TXT=\'{}\'".format(self.student_id,
                                                             self.url_txt))
                print_red("--------------------------------------")
                self.clear()
                return None

    def analyze_result(self):
        self.filter_times.sort()
        self.filter_time = sum(self.filter_times[1:-1]) / 3
        return None


    def execute(self, raw_cmd, filter_idx, iter_idx):
        cmd = raw_cmd.format(filter_idx)
        out, err, retval = execute_command(cmd, sp.PIPE, sp.PIPE)
        out = out + err

        # correctness - execution
        if retval != 0:
            self.filter_correct[filter_idx - 1] = False
            self.filter_correct_score[filter_idx - 1] = 0

            print_red("--------------------------------------")
            print_red("ID=\'{}\' fails executing conv with filter num {}.".format(
                          self.student_id, filter_idx))
            print_red("--------------------------------------")
            return None

        # correctness - speedup
        output = out.decode('utf-8')
        flag = "PASS:\t("
        start = output.find(flag) + len(flag)
        speedup = float(output[start: output[start:].find('x') + start])
        if speedup <= 5:
            self.filter_correct[filter_idx - 1] = False
            self.filter_correct_score[filter_idx - 1] -= 15
            print_red("--------------------------------------")
            print_red("ID=\'{}\' speedup of ./conv -f {} is not greater than 5.".format(
                          self.student_id, filter_idx))
            print_red("--------------------------------------")

            return None

        # performance
        output_list = output.split('\n')
        for line in output_list:
            if line.find('opencl') != -1:
                time = float(line[line.rfind('[') + 1 : line.rfind(']')])
                self.filter_times[iter_idx] += time

        return None

    def record_result_n_rank(self, directory, result_time):
        # score txt format: "{sum of time of ./conv with filter1, 2, 3}"
        with open("{}/{}.txt".format(directory, self.student_id), "w") as file:
            file.write(str(result_time))

        self.rank(directory, result_time)

    def rank(self, score_dir, self_time):
        time_list = []
        for filename in os.listdir(score_dir):
            with open(os.path.join(score_dir, filename), "r") as f:
                time_list.append(float(f.read()))
        time_list.sort()

        print("--------------------------------------")
        print("Fastest 5 programs (sum of time of ./conv -f {1, 2, 3}):")
        for i in time_list[:5]:
            print("{:.4f}\n".format(i))

        perf = 0
        threshold = time_list[0] * 1.5
        if self_time > threshold:
            if self_time < time_list[0] * 2:
                perf = 40
            else:
                perf = 20
        else:
            perf = ((threshold - self_time) / (threshold - time_list[0]))
            perf = perf * 60 + 40
        self.filter_perf = perf
        print("Yours: {:.4f}\n".format(self_time))
        print("Current number of samples at *THIS* node: {}".format(len(time_list)))
        print("--------------------------------------")

    def print_scores(self):
        correct_score = 0.0
        print("--------------------------------------")
        print("Scores:")

        # correctness
        for filter_idx in range(1, 4):
            correct_score += self.filter_correct_score[filter_idx - 1]

        perf_score = 20 * (self.filter_perf / 100)
        print("Correctness score = {}".format(correct_score))
        print("Performance score = {:.2f}".format(perf_score))
        print("--------------------------------------")

    def clear(self):
        try:
            os.chdir(self.pp_wd)
            cmd = "rm -rf {}".format(self.grade_wd)
            execute_command(cmd)
            os.chdir(self.prev_wd)
        except:
            print_red("--------------------------------------")
            print_red("Clean grade directory failed!")
            print_red("ID=\'{}\', ZIP=\'{}\'".format(self.student_id, self.zip_name))
            print_red("--------------------------------------")
            os.chdir(self.prev_wd)
            return None

    def run(self):
        try:
            cmd = "mkdir -m 770 -p {}".format(self.grade_wd)
            execute_command(cmd)
            cmd = "cp -r {}/{} {}".format(self.pp_wd, self.src_tmpl, self.grade_wd)
            execute_command(cmd)
        except:
            print_red("--------------------------------------")
            print_red("Prepare grading directory failed!")
            print_red("ID=\'{}\', ZIP=\'{}\'".format(self.student_id, self.zip_name))
            print_red("--------------------------------------")
            self.clear()
            return None

        try:
            cmd = "cp {}/{} {}".format(self.prev_wd, self.zip_name, self.grade_wd)
            execute_command(cmd)
        except:
            print_red("--------------------------------------")
            print_red("Copy [ID].zip to grade directory failed!")
            print_red("ID=\'{}\', ZIP=\'{}\'".format(self.student_id, self.zip_name))
            print_red("--------------------------------------")
            self.clear()
            return None

        try:
            os.chdir("{}/{}".format(self.grade_wd, self.src_tmpl))
            cmd = "unzip -o ../{}".format(self.zip_name)
            execute_command(cmd)
        except:
            print_red("--------------------------------------")
            print_red("UNZIP failed!")
            print_red("ID=\'{}\', ZIP=\'{}\'".format(self.student_id, self.zip_name))
            print_red("--------------------------------------")
            self.clear()
            return None

        try:
            cmd = "make clean"
            execute_command(cmd)
            cmd = "make"
            _, _, retval = execute_command(cmd)
            if retval != 0:
                raise Exception("BUILD fails")
        except:
            print_red("--------------------------------------")
            print_red("BUILD failed!")
            print_red("ID=\'{}\', ZIP=\'{}\'".format(self.student_id, self.zip_name))
            print_red("--------------------------------------")
            self.clear()
            return None

        # > ==========================
        # > Filter Correctness
        # > ==========================
        for iter_idx in range(5):
            for filter_idx in range(1, 4):
                if not self.filter_correct[filter_idx - 1]:
                    break

                cmd = "./conv -f {}"
                self.execute(cmd, filter_idx, iter_idx)

        if False not in self.filter_correct:
            self.analyze_result()
            self.record_result_n_rank(self.score_dir, self.filter_time)

        self.print_scores()

        self.checkUrl()

        #finish
        self.clear()
        print("ID=\'{}\' done.".format(self.student_id))
        os.chdir(self.prev_wd)

if __name__ == '__main__':
    usr_name = pwd.getpwuid(os.getuid()).pw_name
    single = Single_Test("HW6", usr_name)
    single.checkZIP()
    single.run()