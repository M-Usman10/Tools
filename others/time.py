import timeit
def enumerate_t(iter,total_len=None):
    count =0
    prev = timeit.default_timer()
    for item in iter:
        count+=1
        current_time = timeit.default_timer()
        diff =  current_time - prev
        prev = current_time
        if total_len is not None:
            diff = (total_len-count)*diff
        yield count, diff, item