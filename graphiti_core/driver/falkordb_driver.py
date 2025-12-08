"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import datetime
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from falkordb import Graph as FalkorGraph
    from falkordb.asyncio import FalkorDB
else:
    try:
        from falkordb import Graph as FalkorGraph
        from falkordb.asyncio import FalkorDB
    except ImportError:
        # If falkordb is not installed, raise an ImportError
        raise ImportError(
            'falkordb is required for FalkorDriver. '
            'Install it with: pip install graphiti-core[falkordb]'
        ) from None

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices

logger = logging.getLogger(__name__)

STOPWORDS = [
    'a',
    'is',
    'the',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'but',
    'by',
    'for',
    'if',
    'in',
    'into',
    'it',
    'no',
    'not',
    'of',
    'on',
    'or',
    'such',
    'that',
    'their',
    'then',
    'there',
    'these',
    'they',
    'this',
    'to',
    'was',
    'will',
    'with',
]


class FalkorDriverSession(GraphDriverSession):
    provider = GraphProvider.FALKORDB

    def __init__(self, graph: FalkorGraph, driver: 'FalkorDriver'):
        self.graph = graph
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Falkor, but method must exist
        pass

    async def close(self):
        # No explicit close needed for FalkorDB, but method must exist
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        # Ensure datetime support detection has run
        if self.driver._supports_native_datetime is None:
            await self.driver._initialize_datetime_support()

        # FalkorDB does not support argument for Label Set, so it's converted into an array of queries
        if isinstance(query, list):
            for cypher, params in query:
                # Inject localdatetime() wrappers for datetime parameters
                prepared_query, prepared_params = self.driver._inject_localdatetime_wrappers(
                    str(cypher), params
                )
                await self.graph.query(prepared_query, prepared_params)  # type: ignore[reportUnknownArgumentType]
        else:
            params = dict(kwargs)
            # Inject localdatetime() wrappers for datetime parameters
            prepared_query, prepared_params = self.driver._inject_localdatetime_wrappers(
                str(query), params
            )
            await self.graph.query(prepared_query, prepared_params)  # type: ignore[reportUnknownArgumentType]
        # Assuming `graph.query` is async (ideal); otherwise, wrap in executor
        return None


class FalkorDriver(GraphDriver):
    provider = GraphProvider.FALKORDB
    default_group_id: str = '\\_'
    fulltext_syntax: str = '@'  # FalkorDB uses a redisearch-like syntax for fulltext queries
    aoss_client: None = None

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        username: str | None = None,
        password: str | None = None,
        falkor_db: FalkorDB | None = None,
        database: str = 'default_db',
    ):
        """
        Initialize the FalkorDB driver.

        FalkorDB is a multi-tenant graph database.
        To connect, provide the host and port.
        The default parameters assume a local (on-premises) FalkorDB instance.

        Args:
        host (str): The host where FalkorDB is running.
        port (int): The port on which FalkorDB is listening.
        username (str | None): The username for authentication (if required).
        password (str | None): The password for authentication (if required).
        falkor_db (FalkorDB | None): An existing FalkorDB instance to use instead of creating a new one.
        database (str): The name of the database to connect to. Defaults to 'default_db'.
        """
        super().__init__()
        self._database = database
        if falkor_db is not None:
            # If a FalkorDB instance is provided, use it directly
            self.client = falkor_db
        else:
            self.client = FalkorDB(host=host, port=port, username=username, password=password)

        # Flag to track datetime support (None = not checked yet, True = supported, False = not supported)
        self._supports_native_datetime: bool | None = None

        # Schedule the indices and constraints to be built and datetime detection
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Schedule datetime support detection
            loop.create_task(self._initialize_datetime_support())
            # Schedule the build_indices_and_constraints to run
            loop.create_task(self.build_indices_and_constraints())
        except RuntimeError:
            # No event loop running, this will be handled later
            pass

    def _get_graph(self, graph_name: str | None) -> FalkorGraph:
        # FalkorDB requires a non-None database name for multi-tenant graphs; the default is "default_db"
        if graph_name is None:
            graph_name = self._database
        return self.client.select_graph(graph_name)

    async def _initialize_datetime_support(self):
        """
        Check if FalkorDB supports native datetime on first connection.
        Sets _supports_native_datetime flag for future queries.
        """
        if self._supports_native_datetime is None:
            self._supports_native_datetime = await self._detect_datetime_support()

            if self._supports_native_datetime:
                logger.info('FalkorDB native datetime support: enabled')
                # Check for legacy string dates if native datetime is supported
                await self._check_for_legacy_datetime_strings()
            else:
                logger.warning(
                    'FalkorDB native datetime support: disabled (using string format). '
                    'String-based datetime is deprecated and will be removed in a future version. '
                    'Please upgrade to FalkorDB v4.2.0+ for native datetime support.'
                )

    async def _detect_datetime_support(self) -> bool:
        """
        Test if FalkorDB supports localdatetime() function.

        Returns:
            True if localdatetime() is supported, False otherwise
        """
        try:
            # Try to execute a simple localdatetime() query
            graph = self._get_graph(self._database)
            result = graph.query("RETURN localdatetime('2024-01-01T00:00:00Z') as dt")
            # If query succeeds, localdatetime() is supported
            return True
        except Exception as e:
            error_msg = str(e).lower()
            # Check for errors indicating the function doesn't exist
            if 'unknown function' in error_msg or 'undefined function' in error_msg:
                return False  # localdatetime() not available
            # Other errors might indicate connection issues - log and assume not supported
            logger.warning(f'Error detecting datetime support, assuming not supported: {e}')
            return False

    async def _check_for_legacy_datetime_strings(self):
        """
        Check if database contains legacy string-format datetime values.
        Warns user if migration is needed.

        This runs automatically on initialization when native datetime is supported.
        """
        try:
            # Check for string-type created_at values in Entity using typeOf()
            result = await self.execute_query("""
                MATCH (n:Entity)
                WHERE n.created_at IS NOT NULL AND typeOf(n.created_at) = 'String'
                RETURN count(n) as legacy_count
                LIMIT 1
            """)

            if result:
                records, _, _ = result
                if records and records[0].get('legacy_count', 0) > 0:
                    logger.warning(
                        'Legacy string-format datetime values detected in database. '
                        'Run driver.migrate_string_dates_to_native() to convert old dates to native datetime format. '
                        'String-format support will be removed in a future version.'
                    )
        except Exception as e:
            # Silently fail if query syntax not supported or other issues
            # This is just a helpful warning, not critical
            logger.debug(f'Could not check for legacy datetime strings: {e}')

    async def migrate_string_dates_to_native(self):
        """
        Migrate legacy string-format datetime values to native datetime.

        This function converts old ISO string dates (e.g., '2024-01-01T00:00:00+00:00')
        to native localdatetime values in FalkorDB.

        Migrates datetime fields in:
        - Entity: created_at
        - Episodic: created_at, valid_at
        - EntityEdge: created_at, expired_at, invalid_at

        Usage:
            driver = FalkorDriver(...)
            await driver.migrate_string_dates_to_native()

        Note: This is a one-time migration. Run this after upgrading to FalkorDB
        with native datetime support if you have existing data.
        """
        if not self._supports_native_datetime:
            logger.warning('Native datetime not supported, skipping migration')
            return

        logger.info('Starting migration of legacy string dates to native datetime...')

        migration_queries = [
            # Entity.created_at
            """
            MATCH (n:Entity)
            WHERE n.created_at IS NOT NULL AND typeOf(n.created_at) = 'String'
            SET n.created_at = localdatetime(n.created_at)
            RETURN count(n) as migrated
            """,
            # Episodic.created_at
            """
            MATCH (n:Episodic)
            WHERE n.created_at IS NOT NULL AND typeOf(n.created_at) = 'String'
            SET n.created_at = localdatetime(n.created_at)
            RETURN count(n) as migrated
            """,
            # Episodic.valid_at
            """
            MATCH (n:Episodic)
            WHERE n.valid_at IS NOT NULL AND typeOf(n.valid_at) = 'String'
            SET n.valid_at = localdatetime(n.valid_at)
            RETURN count(n) as migrated
            """,
            # EntityEdge.created_at
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.created_at IS NOT NULL AND typeOf(e.created_at) = 'String'
            SET e.created_at = localdatetime(e.created_at)
            RETURN count(e) as migrated
            """,
            # EntityEdge.expired_at
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.expired_at IS NOT NULL AND typeOf(e.expired_at) = 'String'
            SET e.expired_at = localdatetime(e.expired_at)
            RETURN count(e) as migrated
            """,
            # EntityEdge.invalid_at
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.invalid_at IS NOT NULL AND typeOf(e.invalid_at) = 'String'
            SET e.invalid_at = localdatetime(e.invalid_at)
            RETURN count(e) as migrated
            """,
        ]

        total_migrated = 0
        for query in migration_queries:
            try:
                result = await self.execute_query(query)
                if result:
                    records, _, _ = result
                    migrated = records[0].get('migrated', 0) if records else 0
                    total_migrated += migrated
            except Exception as e:
                logger.error(f'Error during migration: {e}')
                raise

        logger.info(f'Migration complete! Converted {total_migrated} datetime fields to native format.')
        return total_migrated

    def _inject_localdatetime_wrappers(
        self, query: str, params: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert datetime parameters based on FalkorDB version support.

        FalkorDB does not accept Python datetime objects directly as parameters.
        This method handles datetime parameters in two ways:
        - If native datetime supported: Injects localdatetime() wrappers for native temporal storage
        - If not supported: Converts to ISO strings for backward compatibility

        Args:
            query: Cypher query string with $param placeholders
            params: Dictionary of query parameters

        Returns:
            Tuple of (prepared_query, prepared_params) where datetime parameters are processed

        Example (native datetime):
            query: "CREATE (n {dt: $created_at})"
            params: {"created_at": datetime(2024, 1, 1)}

            Returns:
                prepared_query: "CREATE (n {dt: localdatetime('2024-01-01T12:00:00+00:00')})"
                prepared_params: {}  # created_at removed, now in query

        Example (legacy string):
            query: "CREATE (n {dt: $created_at})"
            params: {"created_at": datetime(2024, 1, 1)}

            Returns:
                prepared_query: "CREATE (n {dt: $created_at})"  # UNCHANGED
                prepared_params: {"created_at": "2024-01-01T12:00:00+00:00"}  # datetime → ISO string
        """
        from graphiti_core.utils.datetime_utils import ensure_utc

        import re

        # Determine datetime conversion strategy based on detected support
        use_native_datetime = self._supports_native_datetime if self._supports_native_datetime is not None else True

        prepared_query = query
        prepared_params = {}

        for key, value in params.items():
            placeholder = f'${key}'

            if isinstance(value, datetime.datetime):
                utc_dt = ensure_utc(value)
                if utc_dt is None:
                    prepared_params[key] = None
                    continue

                iso_str = utc_dt.isoformat()

                if use_native_datetime:
                    # NEW: Use localdatetime() wrapper for native temporal type
                    # Check if this parameter is already inside a localdatetime() call
                    # Pattern: localdatetime($key) or localdatetime( $key )
                    escaped_placeholder = re.escape(placeholder)
                    pattern = rf'localdatetime\s*\(\s*{escaped_placeholder}\s*\)'
                    is_wrapped = re.search(pattern, query, re.IGNORECASE)

                    if is_wrapped:
                        # Already wrapped, keep as string parameter
                        prepared_params[key] = iso_str
                    else:
                        # Not wrapped, inject inline
                        replacement = f"localdatetime('{iso_str}')"
                        prepared_query = prepared_query.replace(placeholder, replacement)
                        # Don't include datetime param in prepared params (it's now in the query)
                else:
                    # OLD: Store as ISO string for backward compatibility
                    prepared_params[key] = iso_str
            else:
                # Keep non-datetime params
                prepared_params[key] = value

        return prepared_query, prepared_params

    async def execute_query(self, cypher_query_, **kwargs: Any):
        # Ensure datetime support detection has run
        if self._supports_native_datetime is None:
            await self._initialize_datetime_support()

        graph = self._get_graph(self._database)

        # Convert datetime parameters based on FalkorDB version support
        # Uses native datetime if supported, otherwise falls back to ISO strings
        prepared_query, prepared_params = self._inject_localdatetime_wrappers(
            cypher_query_, dict(kwargs)
        )

        try:
            result = await graph.query(prepared_query, prepared_params)  # type: ignore[reportUnknownArgumentType]
        except Exception as e:
            if 'already indexed' in str(e):
                # check if index already exists
                logger.info(f'Index already exists: {e}')
                return None
            logger.error(f'Error executing FalkorDB query: {e}\n{prepared_query}\n{prepared_params}')
            raise

        # Convert the result header to a list of strings
        header = [h[1] for h in result.header]

        # Convert FalkorDB's result format (list of lists) to the format expected by Graphiti (list of dicts)
        records = []
        for row in result.result_set:
            record = {}
            for i, field_name in enumerate(header):
                if i < len(row):
                    record[field_name] = row[i]
                else:
                    # If there are more fields in header than values in row, set to None
                    record[field_name] = None
            records.append(record)

        return records, header, None

    def session(self, database: str | None = None) -> GraphDriverSession:
        return FalkorDriverSession(self._get_graph(database), self)

    async def close(self) -> None:
        """Close the driver connection."""
        if hasattr(self.client, 'aclose'):
            await self.client.aclose()  # type: ignore[reportUnknownMemberType]
        elif hasattr(self.client.connection, 'aclose'):
            await self.client.connection.aclose()
        elif hasattr(self.client.connection, 'close'):
            await self.client.connection.close()

    async def delete_all_indexes(self) -> None:
        result = await self.execute_query('CALL db.indexes()')
        if not result:
            return

        records, _, _ = result
        drop_tasks = []

        for record in records:
            label = record['label']
            entity_type = record['entitytype']

            for field_name, index_type in record['types'].items():
                if 'RANGE' in index_type:
                    drop_tasks.append(self.execute_query(f'DROP INDEX ON :{label}({field_name})'))
                elif 'FULLTEXT' in index_type:
                    if entity_type == 'NODE':
                        drop_tasks.append(
                            self.execute_query(
                                f'DROP FULLTEXT INDEX FOR (n:{label}) ON (n.{field_name})'
                            )
                        )
                    elif entity_type == 'RELATIONSHIP':
                        drop_tasks.append(
                            self.execute_query(
                                f'DROP FULLTEXT INDEX FOR ()-[e:{label}]-() ON (e.{field_name})'
                            )
                        )

        if drop_tasks:
            await asyncio.gather(*drop_tasks)

    async def build_indices_and_constraints(self, delete_existing=False):
        if delete_existing:
            await self.delete_all_indexes()
        index_queries = get_range_indices(self.provider) + get_fulltext_indices(self.provider)
        for query in index_queries:
            await self.execute_query(query)

    def clone(self, database: str) -> 'GraphDriver':
        """
        Returns a shallow copy of this driver with a different default database.
        Reuses the same connection (e.g. FalkorDB, Neo4j).
        """
        if database == self._database:
            cloned = self
        elif database == self.default_group_id:
            cloned = FalkorDriver(falkor_db=self.client)
        else:
            # Create a new instance of FalkorDriver with the same connection but a different database
            cloned = FalkorDriver(falkor_db=self.client, database=database)

        return cloned

    async def health_check(self) -> None:
        """Check FalkorDB connectivity by running a simple query."""
        try:
            await self.execute_query('MATCH (n) RETURN 1 LIMIT 1')
            return None
        except Exception as e:
            print(f'FalkorDB health check failed: {e}')
            raise

    def sanitize(self, query: str) -> str:
        """
        Replace FalkorDB special characters with whitespace.
        Based on FalkorDB tokenization rules: ,.<>{}[]"':;!@#$%^&*()-+=~
        """
        # FalkorDB separator characters that break text into tokens
        separator_map = str.maketrans(
            {
                ',': ' ',
                '.': ' ',
                '<': ' ',
                '>': ' ',
                '{': ' ',
                '}': ' ',
                '[': ' ',
                ']': ' ',
                '"': ' ',
                "'": ' ',
                ':': ' ',
                ';': ' ',
                '!': ' ',
                '@': ' ',
                '#': ' ',
                '$': ' ',
                '%': ' ',
                '^': ' ',
                '&': ' ',
                '*': ' ',
                '(': ' ',
                ')': ' ',
                '-': ' ',
                '+': ' ',
                '=': ' ',
                '~': ' ',
                '?': ' ',
            }
        )
        sanitized = query.translate(separator_map)
        # Clean up multiple spaces
        sanitized = ' '.join(sanitized.split())
        return sanitized

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        """
        Build a fulltext query string for FalkorDB using RedisSearch syntax.
        FalkorDB uses RedisSearch-like syntax where:
        - Field queries use @ prefix: @field:value
        - Multiple values for same field: (@field:value1|value2)
        - Text search doesn't need @ prefix for content fields
        - AND is implicit with space: (@group_id:value) (text)
        - OR uses pipe within parentheses: (@group_id:value1|value2)
        """
        if group_ids is None or len(group_ids) == 0:
            group_filter = ''
        else:
            group_values = '|'.join(group_ids)
            group_filter = f'(@group_id:{group_values})'

        sanitized_query = self.sanitize(query)

        # Remove stopwords from the sanitized query
        query_words = sanitized_query.split()
        filtered_words = [word for word in query_words if word.lower() not in STOPWORDS]
        sanitized_query = ' | '.join(filtered_words)

        # If the query is too long return no query
        if len(sanitized_query.split(' ')) + len(group_ids or '') >= max_query_length:
            return ''

        full_query = group_filter + ' (' + sanitized_query + ')'

        return full_query
