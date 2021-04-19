python create_samples.py ^
    -metadata ..\data\all\examples.json ^
    -train ..\data\all\init_train.json ^
    -valid ..\data\all\init_valid.json ^
    -test ..\data\all\init_test.json ^
    -total ..\data\all\init_total.json ^
    -size 900 ^
    -ratio 80 10 10 ^
    -cap 1 ^
    --quality percussive
