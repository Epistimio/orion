import React from 'react';
import {
  Header,
  HeaderContainer,
  HeaderName,
  HeaderMenu,
  HeaderNavigation,
  HeaderMenuButton,
  HeaderMenuItem,
  SkipToContent,
} from 'carbon-components-react';
import { Link } from 'react-router-dom';

const TutorialHeader = props => (
  <HeaderContainer
    render={({ isSideNavExpanded, onClickSideNavExpand }) => (
      <Header aria-label="Orion Dashboard">
        <SkipToContent />
        <HeaderMenuButton
          aria-label="Open menu"
          onClick={onClickSideNavExpand}
          isActive={isSideNavExpanded}
        />
        <HeaderName element={Link} to="/" prefix="Oríon" replace>
          Dashboard
        </HeaderName>
        <HeaderNavigation aria-label="Oríon Dashboard">
          <HeaderMenu
            aria-label={
              props.dashboard === 'experiments'
                ? 'experiments (selected)'
                : 'experiments'
            }
            menuLinkName="Experiments">
            <HeaderMenuItem
              title="Go to experiments visualizations"
              element={Link}
              to="/visualizations"
              replace>
              Visualizations
            </HeaderMenuItem>
            <HeaderMenuItem
              title="Go to experiments status"
              element={Link}
              to="/status"
              replace>
              Status
            </HeaderMenuItem>
            <HeaderMenuItem
              title="Go to experiments database"
              element={Link}
              to="/database"
              replace>
              Database
            </HeaderMenuItem>
            <HeaderMenuItem
              title="Go to experiments configuration"
              element={Link}
              to="/configuration"
              replace>
              Configuration
            </HeaderMenuItem>
          </HeaderMenu>
          <HeaderMenu
            aria-label={
              props.dashboard === 'benchmarks'
                ? 'benchmarks (selected)'
                : 'benchmarks'
            }
            menuLinkName="Benchmarks">
            <HeaderMenuItem
              title="Go to benchmarks visualizations"
              element={Link}
              to="/benchmarks/visualizations"
              replace>
              Visualizations
            </HeaderMenuItem>
            <HeaderMenuItem
              title="Go to benchmarks status"
              element={Link}
              to="/benchmarks/status"
              replace>
              Status
            </HeaderMenuItem>
            <HeaderMenuItem
              title="Go to benchmarks database"
              element={Link}
              to="/benchmarks/database"
              replace>
              Database
            </HeaderMenuItem>
            <HeaderMenuItem
              title="Go to benchmarks configuration"
              element={Link}
              to="/benchmarks/configuration"
              replace>
              Configuration
            </HeaderMenuItem>
          </HeaderMenu>
        </HeaderNavigation>
      </Header>
    )}
  />
);

export default TutorialHeader;
