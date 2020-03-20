import timeit
def enumerate_t(iter,total_len=None,beta = 0.97):
    count =0
    prev = timeit.default_timer()
    diff_avg = 0
    for item in iter:
        count+=1
        current_time = timeit.default_timer()
        diff =  current_time - prev
        prev = current_time
        if total_len is not None:
            diff = (total_len-count)*diff
            diff_avg = beta* diff_avg + (1-beta)*diff
        yield count, diff_avg, item