var map, startMarker, endMarker, currentMarker, routePolyline;
var destinationSet = false;
var ws = new WebSocket('ws://192.168.95.127:8080');
var routeCoordinates = [];

ws.onopen = function() {
    ws.send('web_connected');
};

ws.onmessage = function(event) {
    var msg = event.data;

    if (msg.startsWith('location:')) {
        var loc = msg.split(':')[1].split(',');
        var lat = parseFloat(loc[0]);
        var lng = parseFloat(loc[1]);
        var latLng = new Tmapv2.LatLng(lat, lng);

        if (!currentMarker) {
            currentMarker = new Tmapv2.Marker({
                position: latLng,
                map: map,
                iconSize: new Tmapv2.Size(24, 38),
            });
        } else {
            currentMarker.setPosition(latLng);
        }

        if (!startMarker) {
            startMarker = new Tmapv2.Marker({
                position: latLng,
                map: map,
                iconSize: new Tmapv2.Size(24, 38),
            });
            findRoute();
        }

        // Check if the current location is deviating from the route
        if (routeCoordinates.length > 0 && !isLocationOnRoute(latLng, routeCoordinates)) {
            ws.send('alert:deviation');
        }
    } else if (msg.startsWith('camera:')) {
        var cameraData = msg.split(':')[1];
        document.getElementById('camera').src = 'data:image/jpeg;base64,' + cameraData;
        console.log('cam')
    }
};

function initTmap() {
    map = new Tmapv2.Map("map", {
        center: new Tmapv2.LatLng(37.5665, 126.9780),
        width: "100%",
        height: "500px",
        zoom: 12
    });

    map.addListener("click", onMapClick);
}

function onMapClick(event) {
    if (destinationSet) return;
    
    var latLng = event.latLng;
    endMarker = new Tmapv2.Marker({
        position: latLng,
        map: map,
        iconSize: new Tmapv2.Size(24, 38),
    });

    var dest = 'destination:' + latLng.lat() + ',' + latLng.lng();
    ws.send(dest);
    destinationSet = true;
}

function findRoute() {
    var headers = {};
    headers["appKey"] = "SieXPHlrFV2F1jetnSDKKqLHdq2f9F18LLLjFhyg";

    var startX = startMarker.getPosition().lng();
    var startY = startMarker.getPosition().lat();
    var endX = endMarker.getPosition().lng();
    var endY = endMarker.getPosition().lat();

    var url = "https://apis.openapi.sk.com/tmap/routes/pedestrian?version=1&format=json&callback=result";
    var params = {
        startX: startX,
        startY: startY,
        endX: endX,
        endY: endY,
        reqCoordType: "WGS84GEO",
        resCoordType: "EPSG3857",
        startName : "출발지",
        endName : "도착지"
    };

    $.ajax({
        method: "POST",
        headers: headers,
        url: url,
        async: false,
        data: params,
        success: function(response) {
            var resultData = response.features;
            routeCoordinates = [];

            for (var i in resultData) {
                var geometry = resultData[i].geometry;
                if (geometry.type === "LineString") {
                    for (var j in geometry.coordinates) {
                        var latlng = new Tmapv2.Point(geometry.coordinates[j][0], geometry.coordinates[j][1]);
                        var convertPoint = new Tmapv2.Projection.convertEPSG3857ToWGS84GEO(latlng);
                        var convertChange = new Tmapv2.LatLng(convertPoint._lat, convertPoint._lng);
                        routeCoordinates.push(convertChange);
                    }
                }
            }

            if (routePolyline) {
                routePolyline.setMap(null);
            }

            routePolyline = new Tmapv2.Polyline({
                path: routeCoordinates,
                strokeColor: "#FF0000",
                strokeWeight: 6,
                map: map
            });
        },
        error: function(request, status, error) {
            console.log("code:" + request.status + "\n" + "message:" + request.responseText + "\n" + "error:" + error);
        }
    });
}

function isLocationOnRoute(location, route) {
    var tolerance = 30; // Adjust tolerance as needed
    for (var i = 0; i < route.length; i++) {
        if (location.distanceTo(route[i]) < tolerance) {
            return true;
        }
    }
    return false;
}

window.onload = initTmap;
