from config import config

config["layers_size"] = [20, 8, 4, 150]
print("config", config)

class RouteElement(nn.Module):
    def __init__(self, element):
        super().__init__()
        self.ele = element
        self.antenna = []
        self.output = None
        self.potential = []

    def antenna_ready(self):
        if len(self.antenna) == self.ele.input_n:
            return True
        else:
            return False

    def potential_rand_pick(self, layer_n):
        if len(self.potential[layer_n]) == 0:
            return None
        else:
            return self.potential[layer_n].pop(random.randrange(len(self.potential[layer_n])))

    def calculate(self):
        output_list = []

        for are in self.antenna:
            output_list.append(are.output)

        self.output = self.ele(output_list)


class LanModelManual(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            config["vocab_size"], config["vocab_embed_size"])
        self.position_embedding_table = nn.Embedding(
            config["sentence_len"], config["vocab_embed_size"])

        self.input_ln = nn.LayerNorm(config["vocab_embed_size"])

        #
        self.output_vocab_proj = nn.Linear(
            config["output_size"], config["vocab_size"])

        #
        self.routed = False

        # build manually
        layers_size = config["layers_size"]
        layers_antenna_max_size = config["layers_antenna_max_size"]

        # for parameter statics
        self.param_elements = nn.ModuleList([])
        #
        self.layer_elements = []
        for idx, l_size in enumerate(layers_size):
            temp_layer_elements = []
            if idx == 0:
                for x in range(l_size):
                    re = RouteElement(Element(
                        config["vocab_embed_size"], 1, config["output_size"],  config["sentence_len"]))
                    re.to(config["device"])
                    self.param_elements.append(re)
                    temp_layer_elements.append(re)
                self.layer_elements.append(temp_layer_elements)
            else:
                for x in range(l_size):

                    re = RouteElement(Element(config["output_size"], random.randint(
                        1, layers_antenna_max_size[idx]), config["output_size"],  config["sentence_len"]))

                    re.to(config["device"])

                    for ls in layers_size:
                        re.potential.append(list(range(ls)))
                    self.param_elements.append(re)
                    temp_layer_elements.append(re)
                self.layer_elements.append(temp_layer_elements)

    def route(self, input):

        if self.routed == True:
            return

        self.routed = True

        for idx, re_array in enumerate(self.layer_elements):

            if idx != 0:
                for r_ele in re_array:

                    while r_ele.antenna_ready() != True:

                        for l in range(idx):

                            if r_ele.antenna_ready():
                                break

                            if random.randint(1, config["layers_antenna_probability_inv"][l]) == 1:
                                pick_n = r_ele.potential_rand_pick(l)
                                if pick_n != None:
                                    r_ele.antenna.append(
                                        self.layer_elements[l][pick_n])

    def forward(self, out_n, idx, targets=None):
        B, T = idx.shape

        if T != config["sentence_len"]:
            exit("error T!=config['sentence_len']")

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=config["device"]))  # (T,C)
        x = self.input_ln(tok_emb + pos_emb)  # (B,T,C)

        #
        self.route(x)

        #
        for idx, re_array in enumerate(self.layer_elements):

            if idx == 0:
                for r_ele in re_array:
                    r_ele.output = r_ele.ele([x])
            else:
                for r_ele in re_array:
                    r_ele.calculate()

        if out_n == -1:
            # pick one of the output
            rand_output = random.choice(self.layer_elements[-1])
        else:
            rand_output = self.layer_elements[-1][out_n]

        logits = self.output_vocab_proj(rand_output.output)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


##########

output_layer_filter = list(range(config["layers_size"][-1]))

output_layer_score = {}
for x in range(config["layers_size"][-1]):
    output_layer_score[x] = {
        "recent_score": [],
        "recent_avg_score": 0,
    }


def update_output_layer_score(index, score):
    output_layer_score[index]["recent_score"].append(score)
    if len(output_layer_score[index]["recent_score"]) >= 4:
        output_layer_score[index]["recent_score"].pop(0)

    if len(output_layer_score[index]["recent_score"]) > 0:
        avg_score = round(sum(output_layer_score[index]["recent_score"]) / len(
            output_layer_score[index]["recent_score"]), 8)
        output_layer_score[index]['recent_avg_score'] = avg_score


def get_rand_output_index():
    return random.choice(output_layer_filter)


def remove_potentials():

    if len(output_layer_filter) <= 1:
        return

    max_score_index = -1
    max_score = 0
    for score_index, score_item in output_layer_score.items():
        if score_item['recent_avg_score'] > max_score:
            max_score = score_item['recent_avg_score']
            max_score_index = score_index

    if max_score_index != -1:
        output_layer_score.pop(max_score_index)
        output_layer_filter.remove(max_score_index)


# @torch.no_grad()
# def estimate_loss_2():
#     out_mean = {}
#     model.eval()
#     out_losses = {}
#     for split in ['train', 'val']:
#         out_losses[split] = torch.zeros(len(output_layer_filter))
#         for out_idx, out_id in enumerate(output_layer_filter):
#             losses = torch.zeros(config["eval_iters"])
#             for k in range(config["eval_iters"]):
#                 X, Y = get_batch(split)
#                 logits, loss = model(out_id, X, Y)
#                 losses[k] = loss.item()
#             out_losses[split][out_idx] = losses.mean()
#         out_mean[split] = out_losses[split].mean()

#         if split == 'val':
#             val_losses = out_losses[split].tolist()
#             max_index = val_losses.index(max(val_losses))
#             to_remove_val = output_layer_filter[max_index]
#             output_layer_filter.remove(to_remove_val)
#             print("output_layer_filter", output_layer_filter)

#     model.train()
#     return out_mean


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X, Y = get_batch(split)
            logits, loss = model(get_rand_output_index(), X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


###########
model = LanModelManual()
m = model.to(config['device'])
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

# counter = 0
for iter in range(config["max_iters"]):

    # every once in a while evaluate the loss on train and val sets
    if iter % config["eval_interval"] == 0 or iter == config["max_iters"] - 1:
        # counter = counter+1
        # if counter % 1000000 == 0:
        #     remove_potentials()
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        remove_potentials()
        continue

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    output_index = get_rand_output_index()
    logits, loss = model(output_index, xb, yb)

    loss_val = loss.item()
    # print("output_index:", output_index, "loss_val:", loss_val)
    update_output_layer_score(output_index, loss_val)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
