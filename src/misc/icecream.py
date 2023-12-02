
from icecream import ic
from icecream import install as ic_install
from colorama import Fore, Back, Style


def init_icecream():
    ic_install()
    ic.configureOutput(
        prefix="===  ",
        outputFunction=lambda s: print(Fore.YELLOW + Style.BRIGHT + s + " ==="),
    )
