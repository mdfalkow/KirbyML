# KirbyML (WIP)
A machine learning project for the video game Kirby's Dream Land for the Nintendo Game Boy.

## About this Project
The goal of this project is to create a machine learning program that is capable of clearing levels in the video game [Kirby's Dream Land](https://en.wikipedia.org/wiki/Kirby%27s_Dream_Land) for the Nintendo Game Boy.

I decided to pursue this project so I could learn more about machine learning.

Kirby is one of my favorite video game series of all time, and I recently developed an interest in machine learning. I thought this project would be a good starting point and a neat way to include some of my interests. 

## Requirements

This project requires Python 3.

## Instructions

1. Clone this repository.
2. Setup a virtual environment
    1. Install `virtualenv`:
        ```bash
        pip install virtualenv
        ```

    2. Create a new Python 3 virtual environment:
        ```bash
        virtualenv venv -p 3
        ```

    3. Activate your Python environment
        ```bash
        source venv/bin/activate
        ```
        
    4. Install required Python modules
        ```bash
        pip install -r requirements.txt
        ```

3. Obtain a ROM file for *Kirby's Dream Land* for the Nintendo GameBoy. (The checksum is available in `custom_integrations/KirbysDreamLand-GameBoy`. You can use this to verify the file.)

4. Run `main.py`
    ```
    python main.py <ROM_PATH> [OPTIONS]
    ```




## Legal Disclaimer
Kirby&trade; and Kirby's Dream Land&trade; are properties of Nintendo / HAL Laboratory, Inc.. 

Nintendo properties are trademarks of Nintendo. © 2018 Nintendo.
