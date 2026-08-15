# ============================================================================
# E300 V2S — surgical closed-loop overlay for Kaggriculture 1.32.7
# Keeps the exact route untouched in nominal play and intervenes only where
# the original action is economically impossible, provably redundant, or late.
# ============================================================================
from collections import Counter, deque

_V2S_BASE_AGENT = agent
_V2S_PREMIUM = ("MELON", "STRAWBERRY", "MILK", "WOOL")
_V2S_END_STEP = 719
_V2S_ENABLE_SEED_CAP = True
_V2S_ENABLE_SELL_CLAMP = True
_V2S_ENABLE_FLOOR_HOLD = True
_V2S_ENABLE_CASH_RESCUE = True
_V2S_ENABLE_MISSED_HIRE = True
_V2S_ENABLE_TERMINAL_HARVEST = True
_V2S_HINGE_GAIN = 8.0

# Official 1.32.7 curves. Existing ranking functions resolve these globals at
# call time, so this updates the inherited impact model without changing routes.
_MARKET_PARAMS = {
    "WHEAT":      (25, 10000, 400, "sqrt",   0.80, "log",    0.20),
    "CARROT":     (35, 10000, 450, "hinge",  1.00, "sqrt",   0.70),
    "TOMATO":     (60, 10000, 200, "hinge",  0.40, "sqrt",   0.60),
    "STRAWBERRY": (120,10000, 100, "sqrt",   0.70, "linear", 1.60),
    "MELON":      (250,10000, 300, "log",    0.20, "sq",     3.60),
    "EGG":        (50, 10000, 332, "hinge",  0.40, "log",    0.20),
    "MILK":       (160,10000, 122, "sqrt",   0.60, "linear", 1.60),
    "WOOL":       (200,10000, 105, "log",    0.20, "sq",     3.20),
    "FERTILIZER": (100,10000, 200, "linear", 0.40, "linear", 0.40),
}


def _shape(name, value, scale=None):
    value = max(0.0, float(value))
    if name == "linear": return value
    if name == "sq": return value * value
    if name == "sqrt": return math.sqrt(value)
    if name == "log": return math.log1p(value)
    if name == "log10": return math.log10(1.0 + value)
    if name == "hinge":
        scale = max(1e-9, float(scale or 1.0))
        u = value / scale
        return u + _V2S_HINGE_GAIN * max(0.0, u - 1.0) ** 2
    raise ValueError(name)


def _market_price(item, inventory):
    base, equilibrium, scale, below_func, below_target, above_func, above_target = _MARKET_PARAMS[item]
    if inventory < equilibrium:
        amp = below_target * base / _shape(below_func, scale, scale)
        price = base + amp * _shape(below_func, equilibrium - inventory, scale)
    else:
        amp = above_target * base / _shape(above_func, scale, scale)
        price = base - amp * _shape(above_func, inventory - equilibrium, scale)
    return max(_PRICE_FLOOR, int(round(price)))


def _v2s_plan_suffix(actions):
    crops = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    suffix = [{crop: 0 for crop in crops} for _ in range(len(actions) + 1)]
    running = Counter()
    for step in range(len(actions) - 1, -1, -1):
        action = actions[step]
        for order in [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]:
            if isinstance(order, (list, tuple)) and len(order) >= 2 and order[0] == "PLANT" and order[1] in crops:
                running[str(order[1])] += 1
        suffix[step] = {crop: int(running.get(crop, 0)) for crop in crops}
    return suffix


_V2S_SUFFIX = {
    "low": _v2s_plan_suffix(_E279_LOW_ACTIONS),
    "high": _v2s_plan_suffix(_E279_HIGH_ACTIONS),
}


def _v2s_new_state():
    return {
        "last_step": -1,
        "pending": Counter(),
        "price_history": {item: deque(maxlen=96) for item in _SELLABLE},
        "terminal_actor": None,
    }


_V2S_STATE = {0: _v2s_new_state(), 1: _v2s_new_state()}


def _v2s_step(obs):
    explicit = _get(obs, "step")
    if explicit is not None:
        return max(0, int(explicit or 0))
    return max(0, int(_get(obs, "day", 0) or 0) * 24 + int(_get(obs, "hour", 0) or 0))


def _v2s_state(obs):
    seat = _seat(obs)
    step = _v2s_step(obs)
    state = _V2S_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = _v2s_new_state()
        _V2S_STATE[seat] = state
    state["last_step"] = step
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    for item in _SELLABLE:
        try:
            state["price_history"][item].append(max(1.0, float(_get(prices, item, 1) or 1)))
        except (TypeError, ValueError):
            pass
    return state


def _v2s_route():
    return "high" if _ACTIONS is _E279_HIGH_ACTIONS else "low"


def _v2s_current_plant_demand(action):
    demand = Counter()
    for order in [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]:
        if isinstance(order, (list, tuple)) and len(order) >= 2 and order[0] == "PLANT":
            demand[str(order[1])] += 1
    return demand


def _v2s_cap_seed_buys(obs, action, route, step):
    if not _V2S_ENABLE_SEED_CAP:
        return action
    action = _copy_action(action)
    private = _get(obs, "private", {}) or {}
    seeds = Counter({str(k): max(0, int(v or 0)) for k, v in dict(_get(private, "seeds", {}) or {}).items()})
    current_demand = _v2s_current_plant_demand(action)
    remaining_need = Counter()
    suffix_index = min(step + 1, len(_V2S_SUFFIX[route]) - 1)
    for crop in seeds:
        consumed = current_demand[crop] if seeds[crop] >= current_demand[crop] else 0
        missed_current = current_demand[crop] if consumed == 0 and current_demand[crop] else 0
        seeds[crop] -= consumed
        remaining_need[crop] = int(_V2S_SUFFIX[route][suffix_index].get(crop, 0)) + missed_current
    market = []
    for raw in action.get("market", []) or []:
        order = list(raw)
        if len(order) >= 3 and order[0] == "BUY_SEED":
            crop = str(order[1])
            requested = max(0, int(order[2] or 0))
            need = max(0, int(remaining_need.get(crop, 0)) - int(seeds.get(crop, 0)))
            quantity = min(requested, need)
            if quantity <= 0:
                continue
            order[2] = quantity
            seeds[crop] += quantity
        market.append(order)
    action["market"] = market
    return action


def _v2s_clamp_sells(obs, action):
    if not _V2S_ENABLE_SELL_CLAMP:
        return action
    action = _copy_action(action)
    available = Counter(_projected_shed(obs, action))
    market = []
    for raw in action.get("market", []) or []:
        order = list(raw)
        if _is_sell(order):
            item = str(order[1])
            quantity = min(max(0, int(order[2] or 0)), max(0, int(available.get(item, 0))))
            if quantity <= 0:
                continue
            order[2] = quantity
            available[item] -= quantity
        market.append(order)
    action["market"] = market
    return action


def _v2s_fib(index):
    a, b = 1, 1
    for _ in range(max(0, int(index))):
        a, b = b, a + b
    return a


def _v2s_order_cost(obs, order, hire_index):
    if not order:
        return 0.0, hire_index
    op = str(order[0])
    if op == "HIRE":
        return float(_v2s_fib(hire_index)), hire_index + 1
    if op == "BUY_SEED" and len(order) >= 3:
        return float({"WHEAT":10,"CARROT":20,"TOMATO":50,"STRAWBERRY":100,"MELON":80}.get(str(order[1]),0)) * max(0,int(order[2] or 0)), hire_index
    if op == "BUY_PRODUCT" and len(order) >= 3 and str(order[1]) in _MARKET_PARAMS:
        item = str(order[1]); quantity=max(0,int(order[2] or 0))
        inventory = int(_get(_get(_get(obs,"market",{}) or {},"inventory",{}) or {},item,10000) or 10000)
        cost=0.0
        for _ in range(quantity):
            inventory -= 1
            cost += _market_price(item, inventory)
        return cost, hire_index
    return 0.0, hire_index


def _v2s_estimated_sale_revenue(obs, orders):
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    return sum(max(0,int(order[2] or 0))*max(1.0,float(_get(prices,order[1],1) or 1)) for order in orders if _is_sell(order))


def _v2s_cash_rescue(obs, action):
    if not _V2S_ENABLE_CASH_RESCUE:
        return action
    action = _copy_action(action)
    market = [list(order) for order in action.get("market", []) or []]
    farm = _farm(obs, _seat(obs))
    money = float(_get(farm, "money", 0) or 0)
    hire_index = int(_get(farm, "hires_today", 0) or 0)
    critical_cost = 0.0
    for order in market:
        if not order:
            continue
        if order[0] in ("HIRE", "BUY_SEED") or (order[0] == "BUY_PRODUCT" and len(order)>=2 and order[1] == "WHEAT"):
            cost, hire_index = _v2s_order_cost(obs, order, hire_index)
            critical_cost += cost
    planned_revenue = _v2s_estimated_sale_revenue(obs, market)
    deficit = max(0.0, critical_cost + 25.0 - money - planned_revenue)
    if deficit <= 0:
        return action
    projected = Counter(_projected_shed(obs, action))
    already = Counter()
    for order in market:
        if _is_sell(order): already[str(order[1])] += max(0,int(order[2] or 0))
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    candidates=[]
    for item in _SELLABLE:
        reserve = 3 if item == "WHEAT" else 0
        available=max(0,int(projected.get(item,0))-already[item]-reserve)
        price=max(1.0,float(_get(prices,item,1) or 1))
        if available>0: candidates.append((-price,item,available))
    emergency=[]
    for neg_price,item,available in sorted(candidates):
        if deficit<=0: break
        price=-neg_price
        quantity=min(available,max(1,int(math.ceil(deficit/price))))
        emergency.append(["SELL",item,quantity])
        deficit=max(0.0,deficit-quantity*price)
    if not emergency:
        return action
    sell_counter=Counter()
    non_sells=[]
    for order in market:
        if _is_sell(order): sell_counter[str(order[1])] += max(0,int(order[2] or 0))
        else: non_sells.append(order)
    for order in emergency: sell_counter[str(order[1])] += int(order[2])
    sells=[["SELL",item,qty] for item,qty in sell_counter.items() if qty>0]
    sells.sort(key=lambda o: float(_get(prices,o[1],1) or 1), reverse=True)
    critical=[o for o in non_sells if o and (o[0] in ("HIRE","BUY_SEED") or (o[0]=="BUY_PRODUCT" and len(o)>=2 and o[1]=="WHEAT"))]
    optional=[o for o in non_sells if o not in critical]
    action["market"]=(sells+critical+optional)[:10]
    return action


def _v2s_floor_hold(obs, action, state):
    if not _V2S_ENABLE_FLOOR_HOLD:
        return action
    action=_copy_action(action)
    step=_v2s_step(obs); day=int(_get(obs,"day",step//24) or 0)
    projected=Counter(_projected_shed(obs,action)); total=sum(projected.values())
    prices=_get(_get(obs,"market",{}) or {},"prices",{}) or {}
    money=float(_get(_farm(obs,_seat(obs)),"money",0) or 0)
    has_buys=any(order and order[0] != "SELL" for order in action.get("market",[]) or [])
    market=[]; held=Counter()
    for raw in action.get("market",[]) or []:
        order=list(raw)
        if _is_sell(order):
            item=str(order[1]); quantity=max(0,int(order[2] or 0)); price=max(1.0,float(_get(prices,item,1) or 1))
            if item in _V2S_PREMIUM and price<=1 and day<25 and not has_buys and money>=500 and total<=72:
                held[item]+=quantity
                continue
        market.append(order)
    for item,qty in held.items(): state["pending"][item]+=qty
    planned=Counter()
    for order in market:
        if _is_sell(order): planned[str(order[1])]+=max(0,int(order[2] or 0))
    thresholds={"MELON":65,"STRAWBERRY":40,"MILK":35,"WOOL":45}
    for item in _V2S_PREMIUM:
        available=max(0,int(projected.get(item,0))-planned[item])
        state["pending"][item]=min(max(0,int(state["pending"].get(item,0))),available)
        price=max(1.0,float(_get(prices,item,1) or 1))
        release = step>=600 or day>=25 or total>=85 or price>=thresholds[item]
        if release and state["pending"][item]>0:
            qty=min(available,state["pending"][item])
            existing=next((o for o in market if _is_sell(o) and o[1]==item),None)
            if existing is not None: existing[2]=int(existing[2])+qty
            elif len(market)<10: market.append(["SELL",item,qty])
            else: continue
            state["pending"][item]-=qty
    action["market"]=market[:10]
    return action


def _v2s_action_valid_here(obs, actor, order):
    if not order: return False
    op=order[0]
    farm=_farm(obs,_seat(obs)); positions=[_get(farm,"farmer",[0,0]),*list(_get(farm,"hands",[]) or [])]
    if actor>=len(positions): return False
    tile=_tile_at(farm,positions[actor])
    if op=="PASS": return True
    if op=="HARVEST": return isinstance(tile,dict) and int(tile.get("yield_units",0) or 0)>0
    return True


def _v2s_move(position,target):
    x,y=int(position[0]),int(position[1]); tx,ty=int(target[0]),int(target[1])
    if x!=tx: return ["EAST" if tx>x else "WEST"]
    if y!=ty: return ["SOUTH" if ty>y else "NORTH"]
    return ["PASS"]


def _v2s_terminal_harvest(obs, action, state):
    if not _V2S_ENABLE_TERMINAL_HARVEST:
        return action
    step=_v2s_step(obs)
    if step<708: return action
    action=_align_hands(action,obs)
    farm=_farm(obs,_seat(obs)); private=_get(obs,"private",{}) or {}
    positions=[_get(farm,"farmer",[0,0]),*list(_get(farm,"hands",[]) or [])]
    inventories=[dict(v or {}) for v in list(_get(private,"inventories",[]) or [])]
    while len(inventories)<len(positions): inventories.append({})
    orders=[action.get("farmer",["PASS"]),*list(action.get("hands") or [])]
    size=len(_get(farm,"tiles",[]) or []) or 10; access=_shed_access(size)
    actor=state.get("terminal_actor")
    if actor is not None:
        actor=int(actor)
        if actor>=len(positions): state["terminal_actor"]=None
        else:
            carried=sum(max(0,int(inventories[actor].get(item,0) or 0)) for item in _SELLABLE)
            pos=positions[actor]
            target=min(access,key=lambda p:abs(int(pos[0])-p[0])+abs(int(pos[1])-p[1]))
            if carried>0:
                if tuple(map(int,pos)) in access:
                    orders[actor]=["DROP"]
                    for item in _SELLABLE:
                        qty=max(0,int(inventories[actor].get(item,0) or 0))
                        if qty and len(action.get("market",[]))<10:
                            action.setdefault("market",[]).append(["SELL",item,qty])
                    state["terminal_actor"]=None
                else:
                    orders[actor]=_v2s_move(pos,target)
            else:
                state["terminal_actor"]=None
    if state.get("terminal_actor") is None:
        candidates=[]
        day=int(_get(obs,"day",step//24) or 0); prices=_get(_get(obs,"market",{}) or {},"prices",{}) or {}
        for idx,pos in enumerate(positions):
            tile=_tile_at(farm,pos)
            if not (isinstance(tile,dict) and int(tile.get("yield_units",0) or 0)>0): continue
            if tile.get("kind")=="PLANT":
                crop=str(tile.get("crop")); first={"WHEAT":2,"CARROT":2,"TOMATO":8,"STRAWBERRY":10,"MELON":10}.get(crop,99)
                if day-int(tile.get("planted_day",day) or day)<first: continue
                item=crop
            elif tile.get("animal"):
                item={"GOOSE":"EGG","COW":"MILK","SHEEP":"WOOL"}.get(str(tile.get("animal")))
            else: continue
            if not item: continue
            distance=min(abs(int(pos[0])-p[0])+abs(int(pos[1])-p[1]) for p in access)
            remaining=_V2S_END_STEP-step
            if 1+distance+1>remaining: continue
            base=orders[idx] if idx<len(orders) else ["PASS"]
            if not (base[0]=="PASS" or not _v2s_action_valid_here(obs,idx,base) or step>=712): continue
            value=int(tile.get("yield_units",0) or 0)*max(1,float(_get(prices,item,1) or 1))
            candidates.append((-value,distance,idx))
        if candidates:
            _,_,actor=min(candidates)
            orders[actor]=["HARVEST"]
            state["terminal_actor"]=actor
    action["farmer"]=orders[0] if orders else ["PASS"]
    action["hands"]=orders[1:]
    return action


def _v2s_missed_hire(obs, action, route, step):
    if not _V2S_ENABLE_MISSED_HIRE:
        return action
    action=_copy_action(action)
    farm=_farm(obs,_seat(obs)); actual=len(_get(farm,"hands",[]) or [])
    expected=len((_E279_HIGH_ACTIONS if route=="high" else _E279_LOW_ACTIONS)[min(step,719)].get("hands") or [])
    existing=sum(1 for o in action.get("market",[]) or [] if o and o[0]=="HIRE")
    missing=max(0,expected-actual-existing)
    hour=int(_get(obs,"hour",step%24) or 0)
    if missing and hour<=8 and len(action.get("market",[]))<10:
        action["market"].extend([["HIRE"] for _ in range(min(missing,10-len(action["market"])))])
    return action


def _v2s_agent(obs, configuration=None):
    del configuration
    try:
        state=_v2s_state(obs)
        step=_v2s_step(obs)
        action=_copy_action(_V2S_BASE_AGENT(obs))
        route=_v2s_route()
        action=_v2s_cap_seed_buys(obs,action,route,step)
        action=_v2s_clamp_sells(obs,action)
        action=_v2s_floor_hold(obs,action,state)
        action=_v2s_missed_hire(obs,action,route,step)
        action=_v2s_cash_rescue(obs,action)
        action=_v2s_terminal_harvest(obs,action,state)
        action=_v2s_clamp_sells(obs,action)
        return _align_hands(action,obs)
    except Exception:
        return _V2S_BASE_AGENT(obs)


agent=_v2s_agent
kaggriculture_e300_v2s_agent=_v2s_agent
