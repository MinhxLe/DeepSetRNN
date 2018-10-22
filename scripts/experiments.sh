#python src/main.py --model_name BasicRNN --hidden_size 32
#python src/main.py --model_name BasicRNN --hidden_size 64
#python src/main.py --model_name BasicRNN --hidden_size 128
#python src/main.py --model_name BasicRNN --hidden_size 256

#python src/main.py --model_name DeepSetRNN --hidden_size 128
#python src/main.py --model_name DeepSetRNN --hidden_size 64 
#python src/main.py --model_name DeepSetRNN --hidden_size 32 


#python src/main.py --model_name DeepSetRNN --alpha_contrib 0 --validate True
#python src/main.py --model_name DeepSetRNN --alpha_contrib 0.1 --validate True
#python src/main.py --model_name DeepSetRNN --alpha_contrib 0.25 --validate True
#python src/main.py --model_name DeepSetRNN --alpha_contrib 0.5 --validate True
#python src/main.py --model_name DeepSetRNN --alpha_contrib 1 --validate True
python src/main.py --model_name DeepSetRNN --alpha_contrib 1.5 --validate True
python src/main.py --model_name DeepSetRNN --alpha_contrib 2 --validate True
python src/main.py --model_name DeepSetRNN --alpha_contrib 3 --validate True

#python src/main.py --model_name DeepSetRNN --alpha_contrib 8 --validate True
#python src/main.py --model_name DeepSetRNN --alpha_contrib 16 --validate True
#python src/main.py --model_name DeepSetRNN --alpha_contrib 32 --validate True
#python src/main.py --model_name DeepSetRNN --alpha_contrib 64 --validate True
#python src/main.py --model_name DeepSetRNN --alpha_contrib 128 --validate True
