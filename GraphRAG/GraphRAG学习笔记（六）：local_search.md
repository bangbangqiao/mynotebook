# Local Search

## 一、前言

local search是微软开源项目graphrag的query模块所提供的检索方式的一种。其基础的运行命令是：

~~~~python
python -m graphrag query --query "萧炎是谁？" --method local --root ./graphrag_ollama_test
~~~~

当我想细致了解query中的local search是如何实现的时候，我的当时的心中主要有以下几个疑问：

- local search是从什么地方进行检索的？
- local search如何进行检索的？
- local search能检索出来什么东西？

下面我逐个解答整个这些问题。

## 二、local search是从什么地方检索的？

从基础命令行`--root ./graphrag_ollama_test`中可以了解到，local search是从`graphrag_ollama_test`文件夹下检索的。该文件夹下，有`lancedb`文件夹，以及几个以`parquet`结尾的文件。

> `.parquet` 文件是一种**列式存储格式的文件** ，广泛用于大数据处理和分析领域。它非常适合存储结构化数据（如表格数据），尤其是当你只需要访问某些列而不是整行数据时，Parquet 格式可以提供非常高的效率。

我们可以简单地把parquet当成一个表格，但是我们知道RAG中的检索依靠的是语义相似性（向量间的距离），依靠这些.parquet文件是无法完成检索的，所以主要的玄机还是在lancedb文件夹里面。

> 在使用 **lanceDB** 时，它会在磁盘上生成一个文件夹（或目录），这个文件夹代表了一个 **lanceDB 数据库实例的存储位置**, 显然这个lancedb文件夹就是一个轻量级的lanceDB数据库，其下面还有一些子文件夹，如default-entity-description.lance，其代表的就是具体的表。

为了更直观的感受，让我们来看看这个lancedb数据库中都装了些什么东西。

`default-community-full_content.lance`中的内容：

![image-20250529204727426](D:\WorkPlace\Github\mynotebook\GraphRAG\img\image-20250529204727426.png)

`default-entity-description.lance`和`default-text_unit-text.lance`中的内容：

![image-20250529204907432](D:\WorkPlace\Github\mynotebook\GraphRAG\img\image-20250529204907432.png)

从上面的图，可以看到有一列的名字是`vector`，其存储的就是`text`的语义向量，local search阶段的相似性检索靠的就是它。

为了更好地和前面的index模块关联起来，也是我前面没有介绍到的地方，也就是**这个lance文件夹是什么时候生成的？**， 因为index的模块的主要功能不就是从非结构化的段落里面提取实体和关系， 构建成社区，形成一个一个`.parquet`文件嘛。构建向量化数据库的主要的工作是在`generate_text_embedding`工作流里面， 其将`.parquet`中数据存储到lancedb向量化数据库里面，方便后续的检索。

向量数据库的操作主要封装在`graphrag.vector_stores`包，包括`connect`,`load_documents`, `similarity_search_by_vector`等各类方法。和之前介绍的logger包等那些包的逻辑一样，其也是对外提供了一个工厂方法，可以根据参数，创建不同的向量数据库实例，在这里又不得不感叹微软工程师们的代码能力，真的优雅且使用。

~~~python
class VectorStoreFactory:
    """A factory for vector stores.

    Includes a method for users to register a custom vector store implementation.
    """

    vector_store_types: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, vector_store_type: str, vector_store: type):
        """Register a custom vector store implementation."""
        cls.vector_store_types[vector_store_type] = vector_store

    @classmethod
    def create_vector_store(
        cls, vector_store_type: VectorStoreType | str, kwargs: dict
    ) -> BaseVectorStore:
        """Create or get a vector store from the provided type."""
        match vector_store_type:
            case VectorStoreType.LanceDB:
                return LanceDBVectorStore(**kwargs)
            case VectorStoreType.AzureAISearch:
                return AzureAISearchVectorStore(**kwargs)
            case VectorStoreType.CosmosDB:
                return CosmosDBVectorStore(**kwargs)
            case _:
                if vector_store_type in cls.vector_store_types:
                    return cls.vector_store_types[vector_store_type](**kwargs)
                msg = f"Unknown vector store type: {vector_store_type}"
                raise ValueError(msg)
~~~

通过VectorStoreFactory，结合vector_store_type能够创建不同的向量数据库实例，在这里面默认的就是lancedb，下面是LanceDBVectorStore的具体实现，大家主要关注`load_documents`方法`similarity_search_by_vecto`r方法，这个是创建对应的表和进行相似行搜索的主要逻辑，从`load_documents`也能解释上面的图片为什么会有那四列。其中`id`, `text`, `attributes`的内容本身在对应的`.parquet`文件中都由的，`vector`是调用embedding模型得到的，其对应的向量代表的是`text`，也就是对实体的描述。

~~~python
class LanceDBVectorStore(BaseVectorStore):
    """LanceDB vector storage implementation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def connect(self, **kwargs: Any) -> Any:
        """Connect to the vector storage."""
        self.db_connection = lancedb.connect(kwargs["db_uri"])
        if (
            self.collection_name
            and self.collection_name in self.db_connection.table_names()
        ):
            self.document_collection = self.db_connection.open_table(
                self.collection_name
            )

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into vector storage."""
        data = [
            {
                "id": document.id,
                "text": document.text,
                "vector": document.vector,
                "attributes": json.dumps(document.attributes),
            }
            for document in documents
            if document.vector is not None
        ]

        if len(data) == 0:
            data = None

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float64())),
            pa.field("attributes", pa.string()),
        ])
        # NOTE: If modifying the next section of code, ensure that the schema remains the same.
        #       The pyarrow format of the 'vector' field may change if the order of operations is changed
        #       and will break vector search.
        if overwrite:
            if data:
                self.document_collection = self.db_connection.create_table(
                    self.collection_name, data=data, mode="overwrite"
                )
            else:
                self.document_collection = self.db_connection.create_table(
                    self.collection_name, schema=schema, mode="overwrite"
                )
        else:
            # add data to existing table
            self.document_collection = self.db_connection.open_table(
                self.collection_name
            )
            if data:
                self.document_collection.add(data)

    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by id."""
        if len(include_ids) == 0:
            self.query_filter = None
        else:
            if isinstance(include_ids[0], str):
                id_filter = ", ".join([f"'{id}'" for id in include_ids])
                self.query_filter = f"id in ({id_filter})"
            else:
                self.query_filter = (
                    f"id in ({', '.join([str(id) for id in include_ids])})"
                )
        return self.query_filter

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        if self.query_filter:
            docs = (
                self.document_collection.search(
                    query=query_embedding, vector_column_name="vector"
                )
                .where(self.query_filter, prefilter=True)
                .limit(k)
                .to_list()
            )
        else:
            docs = (
                self.document_collection.search(
                    query=query_embedding, vector_column_name="vector"
                )
                .limit(k)
                .to_list()
            )
        return [
            VectorStoreSearchResult(
                document=VectorStoreDocument(
                    id=doc["id"],
                    text=doc["text"],
                    vector=doc["vector"],
                    attributes=json.loads(doc["attributes"]),
                ),
                score=1 - abs(float(doc["_distance"])),
            )
            for doc in docs
        ]

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a similarity search using a given input text."""
        query_embedding = text_embedder(text)
        if query_embedding:
            return self.similarity_search_by_vector(query_embedding, k)
        return []

    def search_by_id(self, id: str) -> VectorStoreDocument:
        """Search for a document by id."""
        doc = (
            self.document_collection.search()
            .where(f"id == '{id}'", prefilter=True)
            .to_list()
        )
        if doc:
            return VectorStoreDocument(
                id=doc[0]["id"],
                text=doc[0]["text"],
                vector=doc[0]["vector"],
                attributes=json.loads(doc[0]["attributes"]),
            )
        return VectorStoreDocument(id=id, text=None, vector=None)
~~~

最后一句话简单地总结一下，local search是在lancedb这个向量化数据库中检索的，而这个向量化数据库是在generate_text_embedding工作流中生成的，里面最重要的就是有一个vector列，其是相似性搜索的关键，这个vetor是调用我们配置的embedding模型转化的。

## 三、local search如何进行检索的？

第二部分主要介绍了local search是在什么地方检索，现在我们来介绍一下local search如何进行检索的，让我们一步一步来。

### 第一步：根据语义相似性检索出相关实体

首先是根据query`萧炎是谁？`从lancedb向量化数据库中检索出有关系的实体。下面两行的初始化比较重要，`get_embedding_store`明确了要在lancedb数据库下的什么表里面进行相似性检索，在这里面是`default-entity-description`表。`read_indexer_entities`将entitiy和communitities的信息融合到一起，方便后面的擦操作，community_level其用来控制entities的范围，即在community_level之下的entity。

~~~python
description_embedding_store = get_embedding_store(
        config_args=vector_store_args,
        embedding_name=entity_description_embedding,
    )
entities_ = read_indexer_entities(entities, communities, community_level)
~~~

![image-20250530101845173](D:\WorkPlace\Github\mynotebook\GraphRAG\img\image-20250530101845173.png)

在明确了要在`default-entity-description`中进行检索，下面可以进行检索了。其逻辑主要在graphrag.query.context_builder.entity_extraction.py的map_query_to_entities函数中。对实体的检索主要发生在similarity_search_by_text里面，实际就是上面LanceDBVectorStore类的similarity_search_by_text方法。其检索返回的结果就是存储在`default-entity-description`的数据。**在这里向量化数据库的使命就结束了，采用向量化数据的目的就是通过语义的相似性检索出与query相关的实体。**

![image-20250529215150968](D:\WorkPlace\Github\mynotebook\GraphRAG\img\image-20250529214246760.png)

但是只有上面的信息还远远不够的，因为在index阶段，我们获得了很多其他的信息，如`relationship`,`text_unit`等，这些信息我们都应该利用起来，这也是graphrag比rag强大的原因之一。

从上面的图上可以看出来，目前只有对entity描述的信息，但是好在其有entity对应的id,通过id就可以和`entities_`中的信息关联起来了，`entities_`中有更加丰富的信息,主要是`community_id`和`text_unit_ids`。

~~~python
for result in search_results:
    if embedding_vectorstore_key == EntityVectorStoreKey.ID and isinstance(
        result.document.id, str
    ):
        matched = get_entity_by_id(all_entities_dict, result.document.id)
    else:
        matched = get_entity_by_key(
            entities=all_entities,
            key=embedding_vectorstore_key,
            value=result.document.id,
        )
    if matched:
        matched_entities.append(matched)
~~~

![image-20250530103713988](D:\WorkPlace\Github\mynotebook\GraphRAG\img\image-20250530103713988.png)

### 第二步：根据实体检索出社区信息。

在确定相关的实体之后，就是获取与其相关的社区信息了。如何获取呢？从上面的图可以看出来，每一个检索出来的实体，都有对应的`community_id`，这个实体对应的社区。获取社区的信息的主要逻辑在 self._build_community_context里面，主要有两个重要的参数

`selected_entities`和`max_context_tokens`,`max_context_tokens`限制了community_token的长度，所以会优先选择重要的社区，社区的重要性如何确定呢？`selected_entities`中entity涉及到的社区次数越多，这个社区就越重要。

~~~python
community_context, community_context_data = self._build_community_context(
    selected_entities=selected_entities,
    max_context_tokens=community_tokens,
    use_community_summary=use_community_summary,
    column_delimiter=column_delimiter,
    include_community_rank=include_community_rank,
    min_community_rank=min_community_rank,
    return_candidate_context=return_candidate_context,
    context_name=community_context_name,
)

~~~

~~~python
for entity in selected_entities:
    # increase count of the community that this entity belongs to
    if entity.community_ids:
        for community_id in entity.community_ids:
            community_matches[community_id] = (
                community_matches.get(community_id, 0) + 1
            )

# sort communities by number of matched entities and rank
selected_communities = [
    self.community_reports[community_id]
    for community_id in community_matches
    if community_id in self.community_reports
]
for community in selected_communities:
    if community.attributes is None:
        community.attributes = {}
    community.attributes["matches"] = community_matches[community.community_id]
selected_communities.sort(
    key=lambda x: (x.attributes["matches"], x.rank),  # type: ignore
    reverse=True,  # type: ignore
)
~~~

![image-20250530105147772](D:\WorkPlace\Github\mynotebook\GraphRAG\img\image-20250530105147772.png)

### 第三步：根据实体信息检索出与其相关的关系

获取到社区报告后，接下来就是获取与这些实体相关的关系了，这也是传统的RAG无法做到的地方，其也有一个`max_context_tokens`的限制，即优先选择关系比较重要的关系。

~~~python
local_context, local_context_data = self._build_local_context(
            selected_entities=selected_entities,
            max_context_tokens=local_tokens,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            include_relationship_weight=include_relationship_weight,
            top_k_relationships=top_k_relationships,
            relationship_ranking_attribute=relationship_ranking_attribute,
            return_candidate_context=return_candidate_context,
            column_delimiter=column_delimiter,
)
~~~

![image-20250530105415315](D:\WorkPlace\Github\mynotebook\GraphRAG\img\image-20250530105415315.png)

### 第四步：根据实体信息检索与其相关的段落

获取到关系之后，下一步就是获取与这些实体相关的文本段落了。同样是优先选择一些比较重要的文本段落。

~~~python
text_unit_context, text_unit_context_data = self._build_text_unit_context(
            selected_entities=selected_entities,
            max_context_tokens=text_unit_tokens,
            return_candidate_context=return_candidate_context,
        )
~~~

![image-20250530110451903](D:\WorkPlace\Github\mynotebook\GraphRAG\img\image-20250530110451903.png)

到此为止，local search已经检索到了其想要的全部信息，下一步就是把它填入提示词模板，结合`query`与大模型进行交互了。

~~~python
search_prompt = self.system_prompt.format(
	context_data=context_result.context_chunks, response_type=self.response_type
)
history_messages = [
	{"role": "system", "content": search_prompt},
]

for callback in self.callbacks:
	callback.on_context(context_result.context_records)

async for response in self.model.achat_stream(
    prompt=query,
    history=history_messages,
    model_parameters=self.model_params,
    ):
        for callback in self.callbacks:
        callback.on_llm_new_token(response)
        yield response

~~~

## 四、local search检索出了什么？

其实第三部分也已经回答了这个问题，我在系统地理一下逻辑。

- 首先通过语义相似性，从lancedb数据库中的`default-entity-description`表中检索出top-k个相关的实体。（lancedb的使命到此为止。）
- 通过这些实体，因为community_tokens的限制，检索出前n个相关的社区信息，这些信息来自community_reports.parquet。
- 通过这些实体，因为local_tokens的限制， 检索出前n个相关的关系信息，这些信息来自relationship.parquet。
- 通过这些实体，因为text_unit_tokens的限制，检索出前n个相关的文本段落，这些信息来自text_units.parquet。

从这些可以看出，graphrag检索出来的信息，是要比传统的rag要多的。



## 五、总结

从检索的过程可以看出来，检索的质量其实和index的质量是息息相关的，因为语义相似性的判断，发生在具体的query：萧炎是谁？和index提取出来的entity：description之间，如果大家细心的话，会发现，很多实体的description部分为空。所以，query的好坏很依赖index的结果。graphrag虽然效果好，但是真的贵。

