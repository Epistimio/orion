/* eslint-disable camelcase */
/* eslint-disable max-len */
/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2020                                           */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */

export const DEFAULT_AUTH_URL = 'http://127.0.0.1:3000';
export const DEFAULT_DLPD_URL = 'http://127.0.0.1:3000';
export const DEFAULT_EDI_URL = 'http://127.0.0.1:3000';
export const DEFAULT_KEYCLOAK_URL = 'http://127.0.0.1:8180';
export const DEFAULT_GRAFANA_URL = 'https://grafana-openshift-monitoring.apps.ma1gpu10.ma.platformlab.ibm.com:20443/d/XjtBlQuZk/job-resource-usage?orgId=1&var-namespace=All&var-user=All';
export const DEFAULT_DEVOPS_URL = 'https://9.111.141.68';

export const DLPD_REST_PATH = 'platform/rest/deeplearning/v1';

export const getHostName = url => {
    var match = url.match(/:\/\/(www[0-9]?\.)?(.[^/:]+)/i);
    if (match != null && match.length > 2 && typeof match[2] === 'string' && match[2].length > 0) {
    return match[2];
    }
    else {
        return null;
    }
}

export const getDomain = url => {
    var hostName = getHostName(url);
    var domain = hostName;
    
    if (hostName != null) {
        var parts = hostName.split('.');
        var len = parts.length;
  
        if (parts != null && parts.length > 1) {
           domain = parts[1];
           var i = 2;
  
           while (i < len) {
            domain = domain  + '.' + parts[i];
            i++;
           }
        }
    }
    
    return domain;
}
  
export const getServiceUrl = (serviceVarValue, urlVarValue, path, defaultValue) => {
    let url = defaultValue;

    if (urlVarValue && urlVarValue != '') {
        url = urlVarValue;
    } else if (serviceVarValue && serviceVarValue != '') {
        let domain = getDomain(window.location.href);
        url = 'https://' + serviceVarValue + '.' + domain;

        if (path && path !== '') {
            url = url + '/' + path;
        }
    }

    return url;
};

export const getAuthRestUrl = () => {
    if (!window._env_) {
        return DEFAULT_AUTH_URL;
    }
    
    let auth_url = getServiceUrl(
        window._env_.REACT_APP_AUTH_REST_SERVICE, 
        window._env_.REACT_APP_AUTH_REST_URL,
        '', DEFAULT_AUTH_URL);
    
    console.log('auth_url is ' + auth_url);

    return auth_url;
};

export const getDlpdRestUrl = () => {
    if (!window._env_) {
        return DEFAULT_DLPD_URL;
    }

    if (!window._env_.REACT_APP_DLPD_REST_SERVICE && 
        !window._env_.REACT_APP_DLPD_REST_URL) {
        return DEFAULT_DLPD_URL;
    }
    
    let dlpd_url = getServiceUrl(
        window._env_.REACT_APP_DLPD_REST_SERVICE, 
        window._env_.REACT_APP_DLPD_REST_URL,
        DLPD_REST_PATH, DEFAULT_DLPD_URL);

    console.log('dlpd_url is ' + dlpd_url);

    return dlpd_url;
};

export const getEdiRestUrl = () => {
    if (!window._env_) {
        return DEFAULT_EDI_URL;
    }
    
    let edi_url = getServiceUrl(
        window._env_.REACT_APP_EDI_REST_SERVICE, 
        window._env_.REACT_APP_EDI_REST_URL,
        '', DEFAULT_EDI_URL);

    console.log('edi_url is ' + edi_url);

    return edi_url;
};

export const getKeycloakUrl = () => {
    if (!window._env_) {
        return DEFAULT_KEYCLOAK_URL;
    }
    
    let urlPath = 'auth/admin/master/console';

    if (window._env_.REACT_APP_AUTH_REALM && window._env_.REACT_APP_AUTH_REALM != '') {
        urlPath = 'auth/admin/' + window._env_.REACT_APP_AUTH_REALM + '/console';
    }
    
    let keycloak_url = getServiceUrl(
        window._env_.REACT_APP_KEYCLOAK_SERVICE, 
        window._env_.REACT_APP_KEYCLOAK_URL,
        urlPath, DEFAULT_KEYCLOAK_URL);

    console.log('keycloak_url is ' + keycloak_url);

    return keycloak_url;
};

export const getGrafanaUrl = () => {
    if (!window._env_) {
        return DEFAULT_GRAFANA_URL;
    }
    
    let grafana_url = getServiceUrl(
        window._env_.REACT_APP_GRAFANA_SERVICE, 
        window._env_.REACT_APP_GRAFANA_URL,
        '', DEFAULT_GRAFANA_URL);

    console.log('grafana_url is ' + grafana_url);

    return grafana_url;
};

export const getDevOpsUrl = () => {
    if (!window._env_) {
        return DEFAULT_DEVOPS_URL;
    }
    
    let devops_url = getServiceUrl(
        window._env_.REACT_APP_DEVOPS_SERVICE, 
        window._env_.REACT_APP_DEVOPS_URL,
        '', DEFAULT_DEVOPS_URL);

    console.log('devops_url is ' + devops_url);

    return devops_url;
};



