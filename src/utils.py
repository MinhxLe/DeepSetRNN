def generate_batch(data, batch_size=500):
    n = len(data)
    for idx in range(0,n, batch_size):
        yield data[idx:idx+batch_size]

