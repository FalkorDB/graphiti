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

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.driver import GraphProvider

try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver, FalkorDriverSession

    HAS_FALKORDB = True
except ImportError:
    FalkorDriver = None
    HAS_FALKORDB = False


class TestFalkorDriver:
    """Comprehensive test suite for FalkorDB driver."""

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB'):
            self.driver = FalkorDriver()
        self.driver.client = self.mock_client

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_init_with_connection_params(self):
        """Test initialization with connection parameters."""
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB') as mock_falkor_db:
            driver = FalkorDriver(
                host='test-host', port=1234, username='test-user', password='test-pass'
            )
            assert driver.provider == GraphProvider.FALKORDB
            mock_falkor_db.assert_called_once_with(
                host='test-host', port=1234, username='test-user', password='test-pass'
            )

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_init_with_falkor_db_instance(self):
        """Test initialization with a FalkorDB instance."""
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB') as mock_falkor_db_class:
            mock_falkor_db = MagicMock()
            driver = FalkorDriver(falkor_db=mock_falkor_db)
            assert driver.provider == GraphProvider.FALKORDB
            assert driver.client is mock_falkor_db
            mock_falkor_db_class.assert_not_called()

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_provider(self):
        """Test driver provider identification."""
        assert self.driver.provider == GraphProvider.FALKORDB

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_get_graph_with_name(self):
        """Test _get_graph with specific graph name."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        result = self.driver._get_graph('test_graph')

        self.mock_client.select_graph.assert_called_once_with('test_graph')
        assert result is mock_graph

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_get_graph_with_none_defaults_to_default_database(self):
        """Test _get_graph with None defaults to default_db."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        result = self.driver._get_graph(None)

        self.mock_client.select_graph.assert_called_once_with('default_db')
        assert result is mock_graph

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_success(self):
        """Test successful query execution."""
        # Set datetime support to skip detection
        self.driver._supports_native_datetime = True

        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_result.header = [('col1', 'column1'), ('col2', 'column2')]
        mock_result.result_set = [['row1col1', 'row1col2']]
        mock_graph.query = AsyncMock(return_value=mock_result)
        self.mock_client.select_graph.return_value = mock_graph

        result = await self.driver.execute_query('MATCH (n) RETURN n', param1='value1')

        mock_graph.query.assert_called_once_with('MATCH (n) RETURN n', {'param1': 'value1'})

        result_set, header, summary = result
        assert result_set == [{'column1': 'row1col1', 'column2': 'row1col2'}]
        assert header == ['column1', 'column2']
        assert summary is None

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_handles_index_already_exists_error(self):
        """Test handling of 'already indexed' error."""
        # Set datetime support to skip detection
        self.driver._supports_native_datetime = True

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception('Index already indexed'))
        self.mock_client.select_graph.return_value = mock_graph

        with patch('graphiti_core.driver.falkordb_driver.logger') as mock_logger:
            result = await self.driver.execute_query('CREATE INDEX ...')

            mock_logger.info.assert_called_once()
            assert result is None

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_propagates_other_exceptions(self):
        """Test that other exceptions are properly propagated."""
        # Set datetime support to skip detection
        self.driver._supports_native_datetime = True

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception('Other error'))
        self.mock_client.select_graph.return_value = mock_graph

        with patch('graphiti_core.driver.falkordb_driver.logger') as mock_logger:
            with pytest.raises(Exception, match='Other error'):
                await self.driver.execute_query('INVALID QUERY')

            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_injects_localdatetime_wrappers(self):
        """Test that datetime objects are injected as localdatetime() calls."""
        # Set datetime support to True (native datetime enabled)
        self.driver._supports_native_datetime = True

        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_result.header = []
        mock_result.result_set = []
        mock_graph.query = AsyncMock(return_value=mock_result)
        self.mock_client.select_graph.return_value = mock_graph

        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        await self.driver.execute_query(
            'CREATE (n:Node) SET n.created_at = $created_at', created_at=test_datetime
        )

        # Verify query was modified to include localdatetime()
        call_args = mock_graph.query.call_args[0]
        query_sent = call_args[0]
        params_sent = call_args[1]

        # Query should contain localdatetime() call
        assert 'localdatetime(' in query_sent
        assert '2024-01-01T12:00:00' in query_sent
        assert '$created_at' not in query_sent  # Placeholder should be replaced

        # Params should NOT contain created_at (it's in the query now)
        assert 'created_at' not in params_sent

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_session_creation(self):
        """Test session creation with specific database."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        session = self.driver.session()

        assert isinstance(session, FalkorDriverSession)
        assert session.graph is mock_graph

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_session_creation_with_none_uses_default_database(self):
        """Test session creation with None uses default database."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        session = self.driver.session()

        assert isinstance(session, FalkorDriverSession)

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_close_calls_connection_close(self):
        """Test driver close method calls connection close."""
        mock_connection = MagicMock()
        mock_connection.close = AsyncMock()
        self.mock_client.connection = mock_connection

        # Ensure hasattr checks work correctly
        del self.mock_client.aclose  # Remove aclose if it exists

        with patch('builtins.hasattr') as mock_hasattr:
            # hasattr(self.client, 'aclose') returns False
            # hasattr(self.client.connection, 'aclose') returns False
            # hasattr(self.client.connection, 'close') returns True
            mock_hasattr.side_effect = lambda obj, attr: (
                attr == 'close' and obj is mock_connection
            )

            await self.driver.close()

        mock_connection.close.assert_called_once()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_delete_all_indexes(self):
        """Test delete_all_indexes method."""
        with patch.object(self.driver, 'execute_query', new_callable=AsyncMock) as mock_execute:
            # Return None to simulate no indexes found
            mock_execute.return_value = None

            await self.driver.delete_all_indexes()

            mock_execute.assert_called_once_with('CALL db.indexes()')


class TestFalkorDriverSession:
    """Test FalkorDB driver session functionality."""

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_graph = MagicMock()
        # Create mock driver with datetime support already detected (to avoid detection query)
        self.mock_driver = MagicMock()
        self.mock_driver._supports_native_datetime = True
        self.session = FalkorDriverSession(self.mock_graph, self.mock_driver)

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_session_async_context_manager(self):
        """Test session can be used as async context manager."""
        async with self.session as s:
            assert s is self.session

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_close_method(self):
        """Test session close method doesn't raise exceptions."""
        await self.session.close()  # Should not raise

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_write_passes_session_and_args(self):
        """Test execute_write method passes session and arguments correctly."""

        async def test_func(session, *args, **kwargs):
            assert session is self.session
            assert args == ('arg1', 'arg2')
            assert kwargs == {'key': 'value'}
            return 'result'

        result = await self.session.execute_write(test_func, 'arg1', 'arg2', key='value')
        assert result == 'result'

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_run_single_query_with_parameters(self):
        """Test running a single query with parameters."""
        self.mock_graph.query = AsyncMock()

        await self.session.run('MATCH (n) RETURN n', param1='value1', param2='value2')

        self.mock_graph.query.assert_called_once_with(
            'MATCH (n) RETURN n', {'param1': 'value1', 'param2': 'value2'}
        )

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_run_multiple_queries_as_list(self):
        """Test running multiple queries passed as list."""
        self.mock_graph.query = AsyncMock()

        queries = [
            ('MATCH (n) RETURN n', {'param1': 'value1'}),
            ('CREATE (n:Node)', {'param2': 'value2'}),
        ]

        await self.session.run(queries)

        assert self.mock_graph.query.call_count == 2
        calls = self.mock_graph.query.call_args_list
        assert calls[0][0] == ('MATCH (n) RETURN n', {'param1': 'value1'})
        assert calls[1][0] == ('CREATE (n:Node)', {'param2': 'value2'})

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_run_injects_localdatetime_wrappers(self):
        """Test that datetime objects are injected as localdatetime() calls."""
        self.mock_graph.query = AsyncMock()
        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        await self.session.run(
            'CREATE (n:Node) SET n.created_at = $created_at', created_at=test_datetime
        )

        self.mock_graph.query.assert_called_once()
        call_args = self.mock_graph.query.call_args[0]
        query_sent = call_args[0]
        params_sent = call_args[1]

        # Query should contain localdatetime() call
        assert 'localdatetime(' in query_sent
        assert '2024-01-01T12:00:00' in query_sent
        assert '$created_at' not in query_sent

        # Params should NOT contain created_at (it's in the query now)
        assert 'created_at' not in params_sent


class TestDatetimeInjection:
    """Test datetime injection functionality."""

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_inject_localdatetime_single_param(self):
        """Test injecting localdatetime() for a single datetime parameter."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        query = 'CREATE (n {created_at: $dt})'
        params = {'dt': test_datetime}

        modified_query, remaining_params = FalkorDriver._inject_localdatetime_wrappers(
            query, params
        )

        # Query should contain localdatetime() call
        assert 'localdatetime(' in modified_query
        assert '2024-01-01T12:00:00' in modified_query
        assert '$dt' not in modified_query

        # Params should NOT contain dt (it's in the query now)
        assert 'dt' not in remaining_params

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_inject_localdatetime_mixed_params(self):
        """Test injecting datetime with other parameter types."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        query = 'CREATE (n {name: $name, created_at: $dt, value: $value})'
        params = {'name': 'test', 'dt': test_datetime, 'value': 42}

        modified_query, remaining_params = FalkorDriver._inject_localdatetime_wrappers(
            query, params
        )

        # Query should contain localdatetime() call for datetime param
        assert 'localdatetime(' in modified_query
        assert '2024-01-01T12:00:00' in modified_query
        assert '$dt' not in modified_query

        # Other params should remain
        assert '$name' in modified_query
        assert '$value' in modified_query
        assert remaining_params == {'name': 'test', 'value': 42}

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_inject_localdatetime_multiple_datetime_params(self):
        """Test injecting multiple datetime parameters."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        dt1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dt2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        query = 'CREATE (n {start: $start_dt, end: $end_dt})'
        params = {'start_dt': dt1, 'end_dt': dt2}

        modified_query, remaining_params = FalkorDriver._inject_localdatetime_wrappers(
            query, params
        )

        # Both datetime calls should be injected
        assert modified_query.count('localdatetime(') == 2
        assert '2024-01-01' in modified_query
        assert '2024-06-01' in modified_query
        assert '$start_dt' not in modified_query
        assert '$end_dt' not in modified_query

        # No remaining params
        assert len(remaining_params) == 0

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_inject_localdatetime_no_datetime_params(self):
        """Test that non-datetime params are not modified."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        query = 'CREATE (n {name: $name, value: $value})'
        params = {'name': 'test', 'value': 42}

        modified_query, remaining_params = FalkorDriver._inject_localdatetime_wrappers(
            query, params
        )

        # Query should be unchanged
        assert modified_query == query

        # Params should be unchanged
        assert remaining_params == params


# Simple integration test
class TestFalkorDriverIntegration:
    """Simple integration test for FalkorDB driver."""

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_basic_integration_with_real_falkordb(self):
        """Basic integration test with real FalkorDB instance."""
        pytest.importorskip('falkordb')

        falkor_host = os.getenv('FALKORDB_HOST', 'localhost')
        falkor_port = int(os.getenv('FALKORDB_PORT', '6379'))

        try:
            driver = FalkorDriver(host=falkor_host, port=falkor_port)

            # Test basic query execution
            result = await driver.execute_query('RETURN 1 as test')
            assert result is not None

            result_set, header, summary = result
            assert header == ['test']
            assert result_set == [{'test': 1}]

            await driver.close()

        except Exception as e:
            pytest.skip(f'FalkorDB not available for integration test: {e}')

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_datetime_native_storage_roundtrip(self):
        """Test datetime is stored natively and supports comparisons."""
        pytest.importorskip('falkordb')

        falkor_host = os.getenv('FALKORDB_HOST', 'localhost')
        falkor_port = int(os.getenv('FALKORDB_PORT', '6379'))

        try:
            driver = FalkorDriver(host=falkor_host, port=falkor_port)

            test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

            # Store node with datetime
            await driver.execute_query(
                "CREATE (n:TestNode {name: 'datetime_test', created_at: $created_at})",
                created_at=test_datetime,
            )

            # Retrieve and verify native datetime
            result = await driver.execute_query(
                "MATCH (n:TestNode {name: 'datetime_test'}) RETURN n.created_at as dt"
            )
            records, _, _ = result

            retrieved_date = records[0]['dt']

            # Should be datetime object
            assert isinstance(retrieved_date, datetime)
            assert retrieved_date.year == 2024
            assert retrieved_date.month == 1

            # Test datetime comparison (only works with native datetime)
            result = await driver.execute_query(
                'MATCH (n:TestNode) '
                "WHERE n.created_at > localdatetime('2023-01-01T00:00:00') "
                'RETURN count(n) as cnt'
            )
            records, _, _ = result
            assert records[0]['cnt'] == 1  # Should find our node

            # Test datetime comparison with different threshold
            result = await driver.execute_query(
                'MATCH (n:TestNode) '
                "WHERE n.created_at > localdatetime('2024-06-01T00:00:00') "
                'RETURN count(n) as cnt'
            )
            records, _, _ = result
            assert records[0]['cnt'] == 0  # Should NOT find our node

            # Cleanup
            await driver.execute_query("MATCH (n:TestNode {name: 'datetime_test'}) DELETE n")
            await driver.close()

        except Exception as e:
            pytest.skip(f'FalkorDB not available for datetime integration test: {e}')
