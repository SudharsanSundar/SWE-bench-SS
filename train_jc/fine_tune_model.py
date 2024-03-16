import together
import pprint

ppr = pprint.PrettyPrinter()
# together.api_key = "b2eb92f209bb76db57849e5cb93c6daa3cd3b67cbc251e37f7e6aa1c22f898d2" # Jess
together.api_key = "146706bab46d0101232eb1519664fb6e53c5be853f85aae3cfa9abd266e9434d" # SS


def check_data_format(path):
    return together.Files.check(file=path)


def upload_data_file(path):
    return together.Files.upload(file=path)


def view_all_uploaded_files():
    ppr.pprint(together.Files.list()['data'])


def start_ft_job(train_file='',
                 model='mistralai/Mistral-7B-Instruct-v0.2',
                 epochs=3,
                 ckpts=1,
                 batch=4,
                 lr=1e-5,
                 suffix='swe-bench-job',
                 wandb=None):
    resp = together.Finetune.create(
        training_file=train_file,
        model=model,
        n_epochs=epochs,
        n_checkpoints=ckpts,
        batch_size=batch,
        learning_rate=lr,
        suffix=suffix,
        wandb_api_key=wandb,
    )

    return resp, resp['id']


def main():
    ft_data_path = './data_for_fting.jsonl'

    # Check data
    resp = check_data_format(ft_data_path)
    print('Data correctly fomatted?', resp['is_check_passed'])
    print('Full check details:')
    ppr.pprint(resp)
    if not resp['is_check_passed']:
        ppr.pprint(resp)
        raise Exception('something is wrong with uploading the file, see above')

    # Upload file to together server
    resp = upload_data_file(ft_data_path)
    if 'is_check_passed' in resp:
        ppr.pprint(resp)
        raise Exception('something is wrong with uploading the file, see above')

    # Start fting
    train_file_id = resp['id']
    resp, ft_id = start_ft_job(train_file=train_file_id,
                               model='mistralai/Mixtral-8x7B-Instruct-v0.1',
                               # epochs=3,
                               # ckpts=1,
                               # batch=4,
                               # lr=1e-5,
                               # suffix='swe-bench-job',
                               # wandb=None,
                               )

    print('FT details:', ft_id)
    ppr.pprint(resp)

    # you can check progress on together website


if __name__ == '__main__':
    main()









