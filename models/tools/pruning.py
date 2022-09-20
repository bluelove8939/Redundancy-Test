import torch
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune


from models.tools.training import train, test
from models.tools.imagenet_utils.training import train as imagenet_train
from models.tools.imagenet_utils.training import validate as imagenet_validate

class PruneModule(object):
    def __init__(self, train_dataset, loss_fn, optimizer, tuning_ratio):
        self.train_dataset = train_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.tuning_ratio = tuning_ratio

    def remove_prune_model(self, module: torch.nn.Module):
        for sub_idx, sub_module in module._modules.items():
            if isinstance(sub_module, torch.nn.Conv2d):
                prune.remove(sub_module, 'weight')
            elif isinstance(sub_module, torch.nn.Module):
                self.remove_prune_model(sub_module)

    def prune_layer(self, model, step):
        for sub_idx, sub_module in model._modules.items():
            if isinstance(sub_module, torch.nn.Conv2d):
                prune.l1_unstructured(sub_module, 'weight', amount=step)
            elif isinstance(sub_module, torch.nn.Module):
                self.prune_layer(sub_module, step)

    def prune_model(self, model, target_amount=0.3, threshold=1, step=0.1, max_iter=5, pass_normal=False, verbose=2):
        print("\nPruning Configs")
        print(f"- target_amount: {target_amount:.4f}")
        print(f"- loss_fn: {self.loss_fn}")
        print(f"- threshold: {threshold}")
        print(f"- step: {step}")
        print(f"- max_iter: {max_iter}")
        print(f"- pass_normal: {pass_normal}\n")

        tuning_dataloader = DataLoader(self.train_dataset)

        if not pass_normal:
            print("\ntesting normal model...")
            normal_acc, normal_avg_loss = test(tuning_dataloader, model, loss_fn=self.loss_fn, verbose=verbose)
            print(f"normal model test result: acc({normal_acc:.4f}) avg_loss({normal_avg_loss:.4f})")
        else:
            print("normal model test passed")
            normal_acc = 100
            normal_avg_loss = 0

        chkpoint = model.state_dict()
        chkpoint_pruning_amount = 0
        chkpoint_acc = 0
        chkpoint_avg_loss = 0
        current_density = 1
        step_cnt = 0

        while True:
            step_cnt += 1
            print(f"\niter: {step_cnt}")
            pruning_step_succeed = False
            current_density *= (1 - step)
            self.prune_layer(model, step)
            current_acc, current_avg_loss = 100, 0

            for _ in range(max_iter):
                train(tuning_dataloader, model, loss_fn=self.loss_fn, optimizer=self.optimizer, verbose=verbose)
                current_acc, current_avg_loss = test(tuning_dataloader, model, loss_fn=self.loss_fn, verbose=verbose)

                if current_acc > normal_acc - threshold:
                    pruning_step_succeed = True
                    break

            if not pruning_step_succeed:
                model.load_state_dict(chkpoint)
                print(f"pruning failed: acc({current_acc:.4f}) avg_loss({current_avg_loss:.4f})")
                print(f"pruning amount: {chkpoint_pruning_amount}")
                return model

            chkpoint = model.state_dict()
            chkpoint_pruning_amount = 1 - current_density
            chkpoint_acc = current_acc
            chkpoint_avg_loss = current_avg_loss
            print(f"check point generated: pamount({chkpoint_pruning_amount:.4f}) acc({chkpoint_acc:.4f}) "
                  f"avg_loss({chkpoint_avg_loss:.4f})\n")

            if round(chkpoint_pruning_amount, 1) >= target_amount:
                break

        model.load_state_dict(chkpoint)
        print(f"pruning succeed: acc({chkpoint_acc}) avg_loss({chkpoint_avg_loss})")
        print(f"pruning_amount: {chkpoint_pruning_amount}")
        return model

    def prune_imagenet_model(self, model, args, target_amount=0.3, threshold=0.5, step=0.1, max_iter=10, pass_normal=False, verbose=2):
        print("\nPruning Configs")
        print(f"- target_amount: {target_amount:.4f}")
        print(f"- loss_fn: {self.loss_fn}")
        print(f"- threshold: {threshold}")
        print(f"- step: {step}")
        print(f"- max_iter: {max_iter}")
        print(f"- pass_normal: {pass_normal}\n")

        # train_dataloader = torch.utils.data.DataLoader(
        #     self.train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        #
        # if not pass_normal:
        #     print("\ntesting normal model...")
        #     normal_acc, normal_avg_loss = imagenet_validate(train_dataloader, model, criterion=self.loss_fn, args=args, at_prune=False)
        #     print(f"normal model test result: acc({normal_acc:.4f}) avg_loss({normal_avg_loss:.4f})")
        # else:
        #     print("normal model test passed")
        #     normal_acc = 100
        #     normal_avg_loss = 0

        chkpoint = model.state_dict()
        chkpoint_pruning_amount = 0
        chkpoint_acc = 0
        chkpoint_avg_loss = 0
        current_density = 1
        step_cnt = 0

        # Generate tuning dataset
        tuning_size = int(len(self.train_dataset) * self.tuning_ratio)
        tuning_dataset, _ = torch.utils.data.random_split(self.train_dataset, lengths=[
            tuning_size, len(self.train_dataset) - tuning_size])

        if args.distributed:
            tuning_sampler = torch.utils.data.distributed.DistributedSampler(tuning_dataset)
        else:
            tuning_sampler = None

        tuning_dataloader = torch.utils.data.DataLoader(
            tuning_dataset, batch_size=args.batch_size, shuffle=(tuning_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=tuning_sampler)

        while True:
            # Pruning setup
            step_cnt += 1
            print(f"\nstep: {step_cnt}")
            pruning_step_succeed = False
            current_density *= (1 - step)



            # Obtain normal accuracy
            current_acc, current_avg_loss = 100, 0
            normal_acc, normal_avg_loss = imagenet_validate(tuning_dataloader, model, criterion=self.loss_fn,
                                                            args=args, at_prune=False,
                                                            pbar_header=f"normal acc")

            # Pruning each layer
            self.prune_layer(model, step)

            for i in range(max_iter):
                imagenet_train(tuning_dataloader, model, criterion=self.loss_fn, epoch=1, optimizer=self.optimizer,
                               args=args, at_prune=False, pbar_header=f"tuning iter: {i}")
                current_acc, current_avg_loss = imagenet_validate(tuning_dataloader, model, criterion=self.loss_fn,
                                                                  args=args, at_prune=False,
                                                                  pbar_header=f"validation:  {i}")

                if current_acc > normal_acc - threshold:
                    pruning_step_succeed = True
                    break

            if not pruning_step_succeed:
                model.load_state_dict(chkpoint)
                print(f"pruning failed: acc({current_acc:.4f}) avg_loss({current_avg_loss:.4f})")
                print(f"pruning amount: {chkpoint_pruning_amount}")
                return model

            chkpoint = model.state_dict()
            chkpoint_pruning_amount = 1 - current_density
            chkpoint_acc = current_acc
            chkpoint_avg_loss = current_avg_loss
            print(f"check point generated: pamount({chkpoint_pruning_amount:.4f}) acc({chkpoint_acc:.4f}) "
                  f"avg_loss({chkpoint_avg_loss:.4f})\n")

            if round(chkpoint_pruning_amount, 1) >= target_amount:
                break

        model.load_state_dict(chkpoint)
        print(f"pruning succeed: acc({chkpoint_acc}) avg_loss({chkpoint_avg_loss})")
        print(f"pruning_amount: {chkpoint_pruning_amount}")
        return model