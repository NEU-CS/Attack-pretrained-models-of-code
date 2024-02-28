

filepath = "sntAndInsertDeadCode"

greedyQueryTimes = 139524
allQueryTimes = 277894
greedy_occ = greedyQueryTimes/allQueryTimes
GaAttack_suc = 3
allsuc = 49

def get_each_query_times():
    greedy_query_times = 0
    with open(filepath+".log") as f:
        for line in f:
            if line.startswith('Greedy query times:'):
                temp_time = int(line[20:-1])
                greedy_query_times += temp_time
            
    print(greedy_query_times)


if __name__ == '__main__':
    get_each_query_times()
    '''
    从中可以看出Ga Attack大约占了总model query次数的50%,但是总攻击成功率只提升了不到6%。
    得到以下结论：
    1. 在model.query代价不大的小模型中,可以加上GA Attack,以提升SAR
    2. 在LLM的攻击中,由于对LLM的query代价非常大,因此不能加上GA Attack。用非常大的query代价去换取一点微弱的SAR的提升。
    对其余两个任务做同样的分析，查看是不是同样的结果。
    '''