from cryptography.fernet import Fernet

key = "l8yXPlowtL_VNYdddiWSnmZdMZd7LAmwJ6k8CbXmiJw="

system_information_e = 'systeminfo.txt'
clipboard_information_e = 'clipboard.txt'
keys_information_e = 'key_log.txt'


encrypted_files = [system_information_e, clipboard_information_e, keys_information_e]
count = 0

for decrypting_files in encrypted_files:

    with open(encrypted_files[count], 'rb') as f:
        data = f.read()

    fernet = Fernet(key)
    decrypted = fernet.decrypt(data)

    with open(encrypted_files[count], 'wb') as f:
        f.write(decrypted)

    count += 1