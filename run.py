import os
import sys
import time
import threading
import itertools
import logging
from colorama import Fore, Style, init

init(autoreset=True)

class ColorfulFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        levelname = record.levelname
        message = super().format(record)
        return f"{self.COLORS.get(levelname, '')}{message}{Style.RESET_ALL}"

handler = logging.StreamHandler()
handler.setFormatter(ColorfulFormatter('%(levelname)s:%(name)s:%(message)s'))
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)

if sys.version_info.major != 3:
    print(f"{Fore.RED}Error: This application requires Python 3.12{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Current Python version: {sys.version_info.major}.{sys.version_info.minor}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Please upgrade to Python 3.12 or higher as some required libraries are only available in this version.{Style.RESET_ALL}")
    sys.exit(1)

def spinner_task(stop_event, message):
    spinner = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])
    while not stop_event.is_set():
        sys.stdout.write(f"\r{Fore.CYAN}{message} {next(spinner)}{Style.RESET_ALL}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f"\r{Fore.GREEN}✓ {message} completed!{' ' * 20}\n{Style.RESET_ALL}")
    sys.stdout.flush()

def display_url_banner(url):
    border = "+" + "-" * 50 + "+"
    padding = "|" + " " * 50 + "|"
    
    url_display = f"http://localhost:7860"
    url_padding = " " * ((50 - len(url_display)) // 2)
    url_line = f"|{url_padding}{Fore.YELLOW}{Style.BRIGHT}{url_display}{Style.RESET_ALL}{url_padding}|"
    
    print("\n")
    print(f"{Fore.CYAN}{border}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{padding}{Style.RESET_ALL}")
    print(url_line)
    print(f"{Fore.CYAN}{padding}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{border}{Style.RESET_ALL}")
    print("\n")

print(f"{Fore.CYAN}Loading modules...{Style.RESET_ALL}")

stop_spinner = threading.Event()
spinner_thread = threading.Thread(target=spinner_task, args=(stop_spinner, "Initializing CSVQuery-RAG system"))
spinner_thread.daemon = True
spinner_thread.start()

from frontend.app import demo

if __name__ == "__main__":
    if not os.path.exists(".env"):
        stop_spinner.set() 
        print(f"{Fore.RED}Warning: .env file not found. Please create one from .env.example{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Copy .env.example to .env and add your OpenAI API key{Style.RESET_ALL}")
        sys.exit(1)
    
    try:
        print(f"\n{Fore.CYAN}CSVQuery-RAG system is starting...{Style.RESET_ALL}")
        
        time.sleep(2)
        
        def monitor_server():
            import socket
            server_ready = False
            max_attempts = 30
            attempts = 0
            
            while not server_ready and attempts < max_attempts:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1)
                        result = s.connect_ex(('127.0.0.1', 7860))
                        if result == 0:
                            server_ready = True
                            stop_spinner.set()
                            time.sleep(0.5)
                            print(f"{Fore.GREEN}CSVQuery-RAG system is ready!{Style.RESET_ALL}")
                            # Display the URL in a big banner
                            display_url_banner("http://localhost:7860")
                        else:
                            time.sleep(1)
                            attempts += 1
                except Exception:
                    time.sleep(1)
                    attempts += 1
            
            if not server_ready:
                stop_spinner.set()
                print(f"{Fore.YELLOW}Waiting for server to be fully ready...{Style.RESET_ALL}")
                display_url_banner("http://localhost:7860")
        
        # Start the monitor thread
        monitor_thread = threading.Thread(target=monitor_server)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        demo.launch(show_error=True, server_name="127.0.0.1", server_port=7860, share=False)
            
    except KeyboardInterrupt:
        stop_spinner.set()
        print(f"\n{Fore.YELLOW}CSVQuery-RAG system shutdown by user{Style.RESET_ALL}")
    except Exception as e:
        stop_spinner.set()
        print(f"\n{Fore.RED}Error starting CSVQuery-RAG system: {str(e)}{Style.RESET_ALL}")
        sys.exit(1) 