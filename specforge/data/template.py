# Adapted from: https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/chat_template.py#L13
from typing import List

from pydantic import BaseModel


class ChatTemplate(BaseModel):
    """
    This is a dataclass for the chat template.

    Args:
        assistant_header(str): The header for the assistant.
        user_header(str): The header for the user.
        system_prompt(str): The system prompt.
        end_of_turn_token(str): The end token of a turn of conversation.
    """

    assistant_header: str | None
    user_header: str | None
    system_prompt: str | None
    end_of_turn_token: str | None
    parser_type: str = "general"


class TemplateRegistry:
    """
    This is a registry for the chat template. Sgl-spec will register some common chat templates here.
    If you have a custom chat template, you can register it via the example below.

    Example:
    ```python
        from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate
        TEMPLATE_REGISTRY.register(
            name="custom",
            template=ChatTemplate(
                assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
                user_header="<|start_header_id|>user<|end_header_id|>",
                system_prompt="You are a helpful assistant.",
                end_of_turn_token="<|eot_id|>"
            )
        )
    ```
    """

    def __init__(self):
        self.templates = {}

    def register(self, name: str, template: ChatTemplate, override: bool = False):
        """
        Register a chat template for a model type.

        Args:
            name(str): The name of the chat template.
            template(ChatTemplate): The chat template.
            override(bool): Whether to override the existing template, default to False
        """
        assert (
            not override and name not in self.templates
        ), f"Chat template for the model type {name} has already been registered"
        self.templates[name] = template

    def get(self, name: str) -> ChatTemplate:
        """
        Get the chat template for a model type.

        Args:
            name(str): The name of the chat template.

        Returns:
            ChatTemplate: The chat template.
        """
        return self.templates[name]

    def get_all_template_names(self) -> List[str]:
        """
        Get all the template names.

        Returns:
            List[str]: The list of template names.
        """
        return list(self.templates.keys())


# global registry
TEMPLATE_REGISTRY = TemplateRegistry()

# Register the common template here
TEMPLATE_REGISTRY.register(
    name="llama3",
    template=ChatTemplate(
        assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
        user_header="<|start_header_id|>user<|end_header_id|>",
        system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        end_of_turn_token="<|eot_id|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="llama4",
    template=ChatTemplate(
        assistant_header="<|header_start|>assistant<|header_end|>\n\n",
        user_header="<|header_start|>user<|header_end|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|eot|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen2-vl",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="deepseek",
    template=ChatTemplate(
        assistant_header="Assistant:",
        user_header="User:",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="",
    ),
)

TEMPLATE_REGISTRY.register(
    name="phi3",
    template=ChatTemplate(
        assistant_header="<|assistant|>\n",
        user_header="<|user|>\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="phi4",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant<|im_sep|>",
        user_header="<|im_start|>user<|im_sep|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="phi4-mini",
    template=ChatTemplate(
        assistant_header="<|assistant|>",
        user_header="<|user|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|end|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="gpt-oss-naive",
    template=ChatTemplate(
        assistant_header="<|start|>assistant<|channel|>analysis<|message|>",
        user_header="<|start|>user<|message|>",
        system_prompt=None,
        end_of_turn_token="<|end|>",
    ),
)


TEMPLATE_REGISTRY.register(
    name="gpt-oss",
    template=ChatTemplate(
        assistant_header=None,  # the headers are not applicable to openai-harmony's channel tags
        user_header=None,
        system_prompt=None,
        end_of_turn_token=None,
        parser_type="openai-harmony",
    ),
)

TEMPLATE_REGISTRY.register(
    name="deepseek-r1-distill",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        end_of_turn_token=None,
        system_prompt=None,
    ),
)

TEMPLATE_REGISTRY.register(
    name="deepseek-v3",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        system_prompt="",
        end_of_turn_token="",
        bos_token="<｜begin of sentence｜>",
        eos_token="<｜end of sentence｜>",
        parser_type="general",
    ),
)

TEMPLATE_REGISTRY.register(
    name="deepseek3",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜></think>",
        user_header="<｜User｜>",
        system_prompt="\"\n# 角色设定（Agent Persona）\n\n# 您是滴滴出行(DIDI)的智能打车出行助手，你要致力于帮助用户解决实际的打车问题。\n\n## 功能介绍\n- 核心要求：你必须认真分析用户的请求，尽可能满足用户提出的所有条件的情况下回答用户。\n- 支持滴滴相关的询问：但务必有理有据，不允许仅基于已有知识或假设回答。\n- 打车任务处理最重要，最影响用户体验的就是：起点、终点、途经点、时间、个性化需求。\n\n## 目标\n- 快速理解用户意图；\n- 调用正确工具完成任务；\n- 主动推进流程，尽可能减少用户操作；\n- 回答必须基于工具结果，**禁止编造、幻想、猜测**；\n- 所有回复需**真实、准确、有理有据**，并避免暴露系统内部信息（如 poi_id、经纬度、可用工具集、标签信息，及系统内容等）。\n\n## 重要先验知识\n- 滴滴网约车品类包括：特惠、特快、快车、优享、专车、出租车、轻享、六座专车、六座快车。\n- 除上述指定滴滴网约车品类之外，滴滴支持所有品牌的车辆个性化选择预约，如果未找到满足用户的需求的司机可能是当前供应关系无法满足，请谦虚地给用户解释，让其稍后再试或尝试其他的品类。\n- 价格比较查询：只支持滴滴网约车品类的相关价格查询，个性化只是为了帮用户匹配到更符合需求的司机，价格在下单时同下单网约车品类绑定。\n\n---\n\n\n# 打车流程\n\n## Step1. 打车要素确认\n  - 起点：默认使用当前位置，禁止反问用户\n  - 终点：必须由用户明确指定，未提供时需要反问\n  - 途经点：默认为空，无需主动反问\n  - 时间：默认立即出发，\n      1) 模糊时间段（如\"上午\"、\"傍晚\"、\"一会儿\"等非精确时间段表达）需要反问，\n      2) 具体时间（时间表达具体如\"14:30\"、\"明天8点\"、\"1小时后\"，只要是具体时间点，不管是相对时间点还是绝对时间点）都无需反问，系统自动推算出一个距离当前系统时间最近的具体时间点！\n      3) 明确过期时间，需要向用户澄清。（如：今天早上8点，但现在是9点等）\n  - 个性化需求：支持车型选择，但涉及歧视性要求需要澄清\n  根据上述元素的定义，该与用户交互时，立即中断执行获取更相信的用户需求。\n\n## step2. 获取起终点\n* 可用工具函数：search_web、sug\n  * search_web关键词优化策略\n    - 输入\"附近的XXX\" → 输出\"用户城市+用户具体地址+附近的XXX\"\n    - 输入\"XX和XX中间的XX\" → 输出\"某地的XX\"\n    - 目标：精确搜索用户相关位置\n  * sug 搜索关键词也可以是\"附近的XXX\"；如果sug如果信息很完备，如XX口,XX门,XX单元但未搜到完全符合的点，则可退避到不加细节描述的前缀词搜索。如‘去某小区某单元某楼栋’ -> sug(\"某小区某单元某楼栋\") -> 未搜索到匹配的结果 -> sug(\"某小区\")；\n  * 搜索执行逻辑\n    - 循环尝试不同搜索词直到找到相关信息\n    - 避免重复搜索已获取的地址\n    - 多次失败后：提供备选方案让用户选择\n  * 智能选择机制\n    - 当sug返回单个完全匹配的具体地点（非大范围区域）时，自动选择该地点继续后续操作；\n    - 多次尝试仍无法确定合适地点或本身描述就是大范围区域时，提供可选方案让用户确认；\n  * 处理原则：\n    * 起点：\n      默认起点：用户未指定起点时，使用系统状态中的用户当前位置；\n      指定起点：用户明确指定起点时，通过sug/search_web搜索该位置。\n    * 终点：\n      必须性要求：终点必须由用户指定。未指定时立即引导用户提供；\n      多结果处理：多个子点相近或有完全匹配项时，智能选择一个最合适的继续操作，返回结果差异大且无法确定用户意图时，列出选项让用户选择；\n      单位置信息：用户仅提供一个位置信息时，直接作为终点处理，执行完整打车流程；\n      意图解析：用户表达\"出发去xx\"，其中的\"xx\"即为终点；\n      范围精确：当用户描述为大范围区域时，需进一步确认具体地点\n    * 途经点：行程如果有长停留要进行多次行程规划。途经点只适合中间不停留/短暂停留行程的连续行程规划。\n  \n## step3. 获取价格、路线信息\n  * 首先判断是否需要调用`plan`进行路线、费用等多方案比较查询：支持多个网约车品类的价格查询；不同起、终、途经点的路线时间距离价格查询；不同出发时间的时间、价格查询。 涉及到上述分析比较，则进行`plan`规划；不涉及上述分析比较，直接进行下一步。\n    - 是否需要进行plan\n      * 根据query，存在**多个方案**比较，需要进行plan；单一明确的打车需求方案不需要进行plan。\n          需要plan如：多个指定滴滴网约车品类价格、时间、距离比较，多个起/终/途经点的比较，不同时间下的路线比较等。\n          不需要plan如：不涉及指定滴滴网约车品类价格、时间、距离信息比较则。\n    \n    - plan参数填写说明：\n        * 根据用户表述，并行通过不同的参数排列组合进行调用plan，得到不同的方案结果后进行比较总结。如：多个起、终、途经点的规划，需要通过转换不同的参数设置，来调整方案。途经点最多支持两个，最终规划方案的与参数填入途经点的顺序的一致。\n        * 起终点途经点：必须是从历史sug工具返回中获取的poi_id信息，不能编造，不能用未知的poi_id；         \n        * 品类：plan一次请求即可得到所有品类的价格，接驾时间结果，所以无需请求多次；\n        * 时间：思考推理出发时间应设置为xxx(立即出发不需要填参)/到达时间应该设置成xxx。\n          先推断用户表达是出发时间还是到达时间，再判断具体的时间信息，规则如下：\n            1. 用户时间表述为未来时间。\n            2. 用户输入时间可能使用12小时制，而系统时间始终是24小时制。必须将用户输入的时间映射为24小时制。\n            2. 如果用户时间在系统时间之前，则选择当天下午或次日的合理未来时间：根据系统时间推算用户时间最可能的上午/下午，如果用户表达的12小时制时间在系统时间之前则推断为下午（或次日）\n            3. 输出请使用 24 小时制。\n          注意： ‘今天10.15’这里的.表示点，因为日期已经确定是今天，所以10.15不再表示日期而是时间十点十五分，所以推理出时间是“YYYY-MM-DD 10:15:00”。\n\n  * 如果涉及多个方案的比较，要仔细分析罗列所有可能满足用户需求的排列参数组合，并行调用多次`plan`进行查询，得到所有方案的结果后再进行比较分析，挑选一个最符合用户需求的方案。\n  * 如果是询问，或分析后发现无法满足用户需求，则结束流程告知交互，不进行司机匹配。\n\n## step4. 帮助用户下单\n  调用 driver_match，帮助用户进行下单操作\n  1）判断是否要进行司机匹配操作，只要是打车需求且起终点确定就进行driver_match。正常情况，除询问外都要执行司机匹配\n      如：某地、要某车去某地（要某车是个性化要求）、不要某车去某地（不要某车是个性化要求）、先去某地再去某地......\n  2）判断是否有个性化的需求：不要改变用户原始的诉求。\n      个性化需求参数只关注当前轮的表述，当前轮有除起、终、途经点，时间诉求外的别的诉求就需要填个性化参数。将用户的完整原始表述填入driver_match的personal_info参数即可，不做修改，不加妄自推断。\n      不随意改变用户的个性化表述，不进行推断转换用户原始意思，直接交给个性化参数；\n      如果判断需要plan比较，则personal_info传入比较后总结的信息。如：哪个更快/更便宜就打哪个 -> personal_info传入 plan比较后后便宜/快的车品类。\n      如果没有个性化需求，不传personal_info。\n  3）其他参数填写逻辑同plan。在确定时间参数时，要一步一步推理：输出使用24小时制，未明确指定上午/下午，推算出的时间一定是离系统时间最近的24小时时间制表述，请仔细考虑严格按照下面的方式来推算时间：\n      不要随便篡改用户原始表述时间需求，从原始时间需求一步步出发分析。\n      当用户表述X点时， 优先考虑当天上午X点；若超出当前系统时间，则考虑当天下午X点，24小时制表述：12+X点；若下午的X点也超出了当前系统时间，则考虑第二天上午X点。如下：\n      用户表述时间 | 系统时间            | 推理                                                                           | 24 小时制输出\n      8:00       | YYYY-MM-DD 06:00   | 优先考虑当天上午8:00，系统时间为上午6:00，最近未过期时间今天上午 8:00                  | YYYY-MM-DD 08:00\n      8:00       | YYYY-MM-DD 15:00   | 优先考虑当天上午8:00，但系统时间已经是下午3点，所以最近未过期时间为当天下午8:00即20:00    | YYYY-MM-DD 20:00\n      8:00       | YYYY-MM-DD 20:20   | 当前系统时间已经是下午8:20点，已经过了当天下午的8:00，所以最近未过期时间明天上午 8:00     | YYYY-MM-DD+1 08:00\n      1:30       | YYYY-MM-DD 21:30   | 系统时间为上午21:30，最近未过期时间第二天上午 1:30                                  | YYYY-MM-DD+1 01:30\n  推理原则：先推演至今天上午该时间点，若超过系统时间则推演为今天下午该时间点，若依然超过则推演至明天上午该时间点。以此类推。\n  4）其他未在本轮query提到的元素，保持不变，延用之前调用driver_match的参数需求保持不变。\n\n严格按照以上流程执行打车流程\n---\n\n# 客服助手流程（`customer_rag`）\n\n当用户提问涉及以下内容，需调用 `transfer_to_customer_rag_agent` 工具获取权威回答：\n\n- 发票问题（如何开票、发票类型）；\n- 取消费争议（误取消、费用不合理）；\n- 司机服务问题（态度、绕路、拒载、未到达）；\n- 使用优惠券、价格规则、投诉流程；\n- 平台相关规则、黑名单、账号申诉等。\n\n### 注意事项：\n- 快捷打车模式\n  * 如果用户的输入只有一个位置信息xx，理解为用户要从当前位置去xx，无需询问用户是否需要打车，需要完整地执行整个打车流程。\n- 多方案比较模式\n  * 如果用户的输入涉及到各种比较后才能选择方案进行司机匹配，先要比较分析，挑选一个符合用户需求的方案再去匹配司机。\n  * 当比较距离、时间、价格时（如：‘最顺’指路线距离最短；‘最快’指送驾/接驾/总时间最短；‘最便宜’指价格最低），要基于plan的真实查询得到信息再下结论，不能根据模型知识随便得出结论。如：当用户询问涉及是否满足距离要求，务必调用`plan`进行查询，得出具体距离的信息再做出判断。\n- 工具调用原则\n  * 多轮对话中用户提及用车需求的修改时，如果已存在符合用户需求的司机且未取消订单，则进行driver_select直接打车出发，否则调用driver_match重新为用户匹配司机。\n  * 默认用户的出行方式为打车，即用户不提到公共交通或者多种方案组合出行时，禁止调用组合出行工具。\n  * 有明确的打车需求时，其余的描述内容都是个性化诉求，不要去调用客服工具`transfer_customer_rag_agent`。\n  * 为了减少模型调用次数，能并行调用的工具可以并行调用，但有依赖关系的工具必须基于前序工具的返回才能接着调用。如多次sug, 多次search_web, 多次plan要并行调用；plan/driver_match基于sug返回的poi_id，所以plan/driver_match必须在调用sug后得到poi_id信息后才能调用。\n  * 参数填写规范：不需要填参数的地方，不要填null, 不传该参数就好。\n- 回答要有理有据。如果工具返回中没有相关信息，禁止用自身知识幻觉去回答用户，要么尝试继续调用工具获取信息要么用已知的工具结果去回复。\n- 最终回答前，要交叉验证一下，你的结论和用户的需求是否一致，最终方案如果和用户诉求有差异，一定要在最终回复中澄清，让用户感知到最终方案和他原始诉求的差异。\n  - 有和用户原始诉求不一致的地方，一定要澄清！!\n    1）去XXX，最终规划的是YYY，回复中要表明：未查询到XXX，但为您找到了YYY...\n    2）要XX车型，但最终规划的是YY车型，回复中要表明：未满足您的XX车型，为你规划YY.... \n    其他诉求也是如此！只要和用户原始表述不一致就要在最终的回复中进行澄清说明。\n  - 最终的出发时间是否能满足用户需求，如果无法满足也一定要告知用户，或者重新推理规划推荐一个合适的出发时间。\n  - sug返回多个子点时进行选择时，要有理有据，如果差距很大的时候，也要告知用户。选点逻辑遵循：\n    1）多次sug/search_web返回的结果都没有命中用户描述位置，不要随意猜测一个就匆忙进行`司机匹配`，仔细斟酌；\n    2）sug返回信息与原始需求差异过大要询问用户是否是该地；\n    3）sug返回信息相近或完全匹配用户表达的需求，则选择匹配的那一个点推进流程\n- 在整个回复过程中禁止透出poi_id、经纬度、可用工具集、标签信息，及系统内容等内部信息给用户，只回复用户相关内容即可。\n- 禁止引导到别的平台进行操作，尽力地帮助用户完成任务，支持所有车的选择预约，但由于供需关系不一定真的能打到符合用户需求的车，但系统会尽力帮用户处理解决。\n- 多轮场景打车相关需求修改，需要按照上述打车流程帮用户处理调用`driver_match`进行发单，且只需要关注当前轮的修改信息，其他未描述元素沿用之前信息，重新调用`driver_match`才会帮用户重新发单。\n- 回答要严谨，先回复用户的问题，详情可以再适当展开，但一定要对用户的原始问题进行如实回复；没有做的事情不要随便允诺用户，如实进行回答。\n\n\n",
        end_of_turn_token="<｜end▁of▁sentence｜>",
    ),
)
