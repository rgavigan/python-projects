MENU = {
    "espresso": {
        "ingredients": {
            "water": 50,
            "coffee": 18,
        },
        "cost": 1.5,
    },
    "latte": {
        "ingredients": {
            "water": 200,
            "milk": 150,
            "coffee": 24,
        },
        "cost": 2.5,
    },
    "cappuccino": {
        "ingredients": {
            "water": 250,
            "milk": 100,
            "coffee": 24,
        },
        "cost": 3.0,
    }
}

resources = {
    "water": 300,
    "milk": 200,
    "coffee": 100,
    "money": 0,
}

# Prompt user
while True:
    drink = input("What would you like? (espresso/latte/cappuccino/report/off): ")
    # Off Switch for Machine
    if drink == 'off':
        break
    # Report resources
    elif drink == 'report':
        for i in resources:
            print(f'{i}: {resources[i]}')

    # Check sufficiency of resources for all 3 drinks
    elif drink == 'espresso':
        if resources["water"] < MENU["espresso"]["ingredients"]["water"]:
            print("Sorry there is not enough water.")
        elif resources["coffee"] < MENU["espresso"]["ingredients"]["coffee"]:
            print("Sorry there is not enough coffee.")
        # Process coins if sufficient
        else:
            quarters = int(input("How many quarters: "))
            dimes = int(input("How many dimes: "))
            nickels = int(input("How many nickels: "))
            pennies = int(input("How many pennies: "))
            totalCoins = (quarters * 0.25) + (dimes * 0.10) + (nickels * 0.05) + (pennies * 0.01)
            totalCoins = round(totalCoins, 2)
            print(f'Money Inserted: ${totalCoins}')

            # Check transaction success
            if totalCoins < MENU["espresso"]["cost"]:
                print("Sorry that's not enough money. Money refunded.")
            else:
                # Update money in machine with cost of espresso and return change
                resources["money"] += MENU["espresso"]["cost"]
                if totalCoins > MENU["espresso"]["cost"]:
                    coinChange = totalCoins - MENU["espresso"]["cost"]
                    coinChange = round(coinChange, 2)
                    print(f"Here is ${coinChange} dollars in change")

                # Make coffee and change coffee machine resources
                resources["water"] -= MENU["espresso"]["ingredients"]["water"]
                resources["coffee"] -= MENU["espresso"]["ingredients"]["coffee"]
                print("Here is your espresso. Enjoy!")

    elif drink == 'latte':
        if resources["water"] < MENU["latte"]["ingredients"]["water"]:
            print("Sorry there is not enough water.")
        elif resources["coffee"] < MENU["latte"]["ingredients"]["coffee"]:
            print("Sorry there is not enough coffee.")
        elif resources["milk"] < MENU["latte"]["ingredients"]["milk"]:
            print("Sorry there is not enough milk")
        # Process coins if sufficient
        else:
            quarters = int(input("How many quarters: "))
            dimes = int(input("How many dimes: "))
            nickels = int(input("How many nickels: "))
            pennies = int(input("How many pennies: "))
            totalCoins = (quarters * 0.25) + (dimes * 0.10) + (nickels * 0.05) + (pennies * 0.01)
            totalCoins = round(totalCoins, 2)
            print(f'Money Inserted: ${totalCoins}')

            # Check transaction success
            if totalCoins < MENU["latte"]["cost"]:
                print("Sorry that's not enough money. Money refunded.")
            else:
                # Update money in machine with cost of espresso and return change
                resources["money"] += MENU["latte"]["cost"]
                if totalCoins > MENU["latte"]["cost"]:
                    coinChange = totalCoins - MENU["latte"]["cost"]
                    coinChange = round(coinChange, 2)
                    print(f"Here is ${coinChange} dollars in change")

                # Make coffee and change coffee machine resources
                resources["water"] -= MENU["latte"]["ingredients"]["water"]
                resources["coffee"] -= MENU["latte"]["ingredients"]["coffee"]
                resources["milk"] -= MENU["latte"]["ingredients"]["milk"]
                print("Here is your latte. Enjoy!")

    elif drink == 'cappuccino':
        if resources["water"] < MENU["cappuccino"]["ingredients"]["water"]:
            print("Sorry there is not enough water.")
        elif resources["coffee"] < MENU["cappuccino"]["ingredients"]["coffee"]:
            print("Sorry there is not enough coffee.")
        elif resources["milk"] < MENU["cappuccino"]["ingredients"]["milk"]:
            print("Sorry there is not enough milk")
        # Process coins if sufficient
        else:
            quarters = int(input("How many quarters: "))
            dimes = int(input("How many dimes: "))
            nickels = int(input("How many nickels: "))
            pennies = int(input("How many pennies: "))
            totalCoins = (quarters * 0.25) + (dimes * 0.10) + (nickels * 0.05) + (pennies * 0.01)
            totalCoins = round(totalCoins, 2)
            print(f'Money Inserted: ${totalCoins}')

            # Check transaction success
            if totalCoins < MENU["cappuccino"]["cost"]:
                print("Sorry that's not enough money. Money refunded.")
            else:
                # Update money in machine with cost of espresso and return change
                resources["money"] += MENU["cappuccino"]["cost"]
                if totalCoins > MENU["cappuccino"]["cost"]:
                    coinChange = totalCoins - MENU["cappuccino"]["cost"]
                    coinChange = round(coinChange, 2)
                    print(f"Here is ${coinChange} dollars in change")

                # Make coffee and change coffee machine resources
                resources["water"] -= MENU["cappuccino"]["ingredients"]["water"]
                resources["coffee"] -= MENU["cappuccino"]["ingredients"]["coffee"]
                resources["milk"] -= MENU["cappuccino"]["ingredients"]["milk"]
                print("Here is your cappuccino. Enjoy!")