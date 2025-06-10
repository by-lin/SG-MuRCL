def get_feats(feat_list, cluster_list, action_sequence, feat_size):
    """Extract features based on action sequence for patch selection."""
    batch_feats = []
    for i, (feat, cluster) in enumerate(zip(feat_list, cluster_list)):
        # Get selected patches based on action sequence
        selected_bag, _ = get_selected_bag_and_graph(
            feat.squeeze(0), 
            cluster, 
            action_sequence[i], 
            feat_size
        )
        batch_feats.append(selected_bag.unsqueeze(0))
    return torch.cat(batch_feats, dim=0)


def train_SMTABMIL(args, epoch, train_set, model, fc, ppo, memory, criterion, optimizer, scheduler):
    """Train function for SmTransformerSmABMIL model."""
    print(f"train_SMTABMIL...")
    length = len(train_set)
    train_set.shuffle()

    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    if args.train_stage == 2:
        model.eval()
        fc.eval()
    else:
        model.train()
        fc.train()

    progress_bar = tqdm(range(args.num_data))
    feat_list, cluster_list, label_list, adj_list, step = [], [], [], [], 0
    batch_idx = 0
    labels_list, outputs_list = [], []
    
    for data_idx in progress_bar:
        loss_list = []

        # Get data from dataset
        data = train_set[data_idx % length]
        if len(data) == 5:  # With adjacency matrix
            feat, cluster, label, case_id, adj_mat = data
            adj_list.append(adj_mat)
        else:  # Without adjacency matrix
            feat, cluster, label, case_id = data
            adj_list.append(None)
        
        assert len(feat.shape) == 2, f"{feat.shape}"
        feat = feat.unsqueeze(0).to(args.device)
        label = label.unsqueeze(0).to(args.device)

        feat_list.append(feat)
        cluster_list.append(cluster)
        label_list.append(label)

        step += 1
        if step == args.batch_size or data_idx == args.num_data - 1:
            labels = torch.cat(label_list)
            action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
            
            # Get selected features and graphs
            feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)
            
            # Handle graph data for GAT
            if args.graph_encoder_type == "gat" and any(adj is not None for adj in adj_list):
                # Process adjacency matrices for batched GAT
                batch_adj_list = []
                for i, adj in enumerate(adj_list):
                    if adj is not None:
                        batch_adj_list.append(adj.to(args.device))
                    else:
                        # Create identity matrix if no adjacency matrix
                        feat_len = feat_list[i].shape[1]
                        batch_adj_list.append(torch.eye(feat_len).to(args.device))
                
                # Forward pass with graph data
                if args.train_model_prime and args.train_stage != 2:
                    outputs, states = model(feats, batch_adj_list)
                    outputs = fc(outputs, restart=True)
                else:
                    with torch.no_grad():
                        outputs, states = model(feats, batch_adj_list)
                        outputs = fc(outputs, restart=True)
            else:
                # Standard forward pass without graph
                if args.train_model_prime and args.train_stage != 2:
                    outputs, states = model(feats)
                    outputs = fc(outputs, restart=True)
                else:
                    with torch.no_grad():
                        outputs, states = model(feats)
                        outputs = fc(outputs, restart=True)

            loss = criterion(outputs, labels)
            loss_list.append(loss)

            losses[0].update(loss.data.item(), len(feat_list))
            acc = accuracy(outputs, labels, topk=(1,))[0]
            top1[0].update(acc.item(), len(feat_list))

            # RL Training Loop
            confidence_last = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
            for patch_step in range(1, args.T):
                if args.train_stage == 1:
                    action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
                else:
                    if patch_step == 1:
                        action_sequence = ppo.select_action(states.to(0), memory, restart_batch=True)
                    else:
                        action_sequence = ppo.select_action(states.to(0), memory)
                
                feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)

                # Forward pass for RL steps
                if args.graph_encoder_type == "gat" and any(adj is not None for adj in adj_list):
                    if args.train_stage != 2:
                        outputs, states = model(feats, batch_adj_list)
                        outputs = fc(outputs, restart=False)
                    else:
                        with torch.no_grad():
                            outputs, states = model(feats, batch_adj_list)
                            outputs = fc(outputs, restart=False)
                else:
                    if args.train_stage != 2:
                        outputs, states = model(feats)
                        outputs = fc(outputs, restart=False)
                    else:
                        with torch.no_grad():
                            outputs, states = model(feats)
                            outputs = fc(outputs, restart=False)
                
                loss = criterion(outputs, labels)
                loss_list.append(loss)
                losses[patch_step].update(loss.data.item(), len(feat_list))

                acc = accuracy(outputs, labels, topk=(1,))[0]
                top1[patch_step].update(acc.item())

                confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
                reward = confidence - confidence_last
                confidence_last = confidence

                reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                memory.rewards.append(reward)

            # Backward pass and optimization
            loss = sum(loss_list) / args.T
            if args.train_stage != 2:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                ppo.update(memory)
            memory.clear_memory()
            torch.cuda.empty_cache()

            labels_list.append(labels.detach())
            outputs_list.append(outputs.detach())

            feat_list, cluster_list, label_list, adj_list, step = [], [], [], [], 0
            batch_idx += 1
            progress_bar.set_description(
                f"Train Epoch: {epoch + 1:2}/{args.epochs:2}. Iter: {batch_idx:3}/{args.eval_step:3}. "
                f"Loss: {losses[-1].avg:.4f}. Acc: {top1[-1].avg:.4f}"
            )
            progress_bar.update()

    progress_bar.close()
    if args.train_stage != 2 and scheduler is not None and epoch >= args.warmup:
        scheduler.step()

    labels = torch.cat(labels_list)
    outputs = torch.cat(outputs_list)
    acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)

    return losses[-1].avg, acc, auc, precision, recall, f1_score


def test_SMTABMIL(args, test_set, model, fc, ppo, memory, criterion):
    """Test function for SmTransformerSmABMIL model."""
    losses = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]
    model.eval()
    fc.eval()
    
    with torch.no_grad():
        feat_list, cluster_list, label_list, case_id_list, adj_list, step = [], [], [], [], [], 0
        
        for data_idx, data in enumerate(test_set):
            if len(data) == 5:  # With adjacency matrix
                feat, cluster, label, case_id, adj_mat = data
                adj_list.append(adj_mat)
            else:  # Without adjacency matrix
                feat, cluster, label, case_id = data
                adj_list.append(None)
            
            feat = feat.unsqueeze(0).to(args.device)
            label = label.unsqueeze(0).to(args.device)
            feat_list.append(feat)
            cluster_list.append(cluster)
            label_list.append(label)
            case_id_list.append(case_id)

        loss_list = []
        labels = torch.cat(label_list)
        action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
        feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)

        # Handle graph data for GAT
        if args.graph_encoder_type == "gat" and any(adj is not None for adj in adj_list):
            batch_adj_list = []
            for i, adj in enumerate(adj_list):
                if adj is not None:
                    batch_adj_list.append(adj.to(args.device))
                else:
                    feat_len = feat_list[i].shape[1]
                    batch_adj_list.append(torch.eye(feat_len).to(args.device))
            
            outputs, states = model(feats, batch_adj_list)
        else:
            outputs, states = model(feats)
        
        outputs = fc(outputs, restart=True)
        loss = criterion(outputs, labels)
        loss_list.append(loss)
        losses[0].update(loss.data.item(), outputs.shape[0])

        confidence_last = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
        
        for patch_step in range(1, args.T):
            if args.train_stage == 1:
                action = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
            else:
                if patch_step == 1:
                    action = ppo.select_action(states.to(0), memory, restart_batch=True)
                else:
                    action = ppo.select_action(states.to(0), memory)
            
            feats = get_feats(feat_list, cluster_list, action_sequence=action, feat_size=args.feat_size)
            
            if args.graph_encoder_type == "gat" and any(adj is not None for adj in adj_list):
                outputs, states = model(feats, batch_adj_list)
            else:
                outputs, states = model(feats)
            
            outputs = fc(outputs, restart=False)
            loss = criterion(outputs, labels)
            loss_list.append(loss)
            losses[patch_step].update(loss.data.item(), len(feat_list))

            confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
            reward = confidence - confidence_last
            confidence_last = confidence

            reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
            memory.rewards.append(reward)
        
        memory.clear_memory()
        acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)

    return losses[-1].avg, acc, auc, precision, recall, f1_score, outputs, labels, case_id_list