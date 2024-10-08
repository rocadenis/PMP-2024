import random

initial_red = 3
initial_blue = 4
initial_black =  2

prime_numbers = [1,3,5]

def experiment(n_simulations):
    red_count = 0 
    blue_count = 0
    black_count = 0


    for _ in range(n_simulations):
        urn = {'red' : initial_red, 'blue' : initial_blue, 'black' : initial_black}

        dice_roll = random.randint(1,6)
        if dice_roll in prime_numbers :
            urn['black'] +=1

        elif dice_roll == 6 :
            urn['red'] +=1
        else :
            urn['blue'] +=1

        total_balls = sum(urn.values())
        draw =  random.choices(['red','blue','black'], weights = urn.values(), k = 1)[0]

    #red probability 
        if draw == 'red':
            red_count +=1
        elif draw == 'blue':
            blue_count +=1
        else :
            black_count +=1

    print(f"Blue balls: {blue_count} and black balls: {black_count}")
    return red_count/n_simulations
 
n_simulations = 1000
estimated_probability = experiment (n_simulations)
print(f"Estimated probability of drawing a red ball: {estimated_probability:.4f}")